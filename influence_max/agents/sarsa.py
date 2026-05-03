"""
SARSA agent for influence maximization (on-policy TD learning).

Algorithm
---------
Standard SARSA(0) update rule:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
where a' is the action that the agent will *actually* take in s' under the
current epsilon-greedy policy. This is the on-policy contrast to DQN's
max_{a'} Q(s', a').

Function approximation
----------------------
Linear: Q(s, a) = w . phi(s, a)

phi(s, a) is a fixed feature map built from learned GNN embeddings:
    phi(s, a) = [ mean(H[seed_set]),                   # state aggregate
                  H[a],                                # candidate embedding
                  mean(H[seed_set]) * H[a],            # Hadamard interaction
                  budget_remaining / budget_max,
                  1.0 ]                                # bias

Total feature dim = 3 * D + 2.

Why linear & on-policy?
- Trains in seconds even on Mac CPU/MPS.
- Learns a transparent weight vector (can be inspected for interpretability).
- On-policy SARSA is more conservative than DQN — it accounts for the
  exploration noise of its own policy, often producing safer strategies.

Online updates (no replay buffer)
- Stores the previous (s, a) and computes the TD update on the next call,
  once it knows what action a' the policy is about to take in s'.
- Final transition uses Q-target = r (terminal — no bootstrapping).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig, resolve_device
from influence_max.environment import State, Transition

logger = logging.getLogger(__name__)


class SARSAAgent(BaseAgent):
    """
    Linear-FA SARSA agent with on-policy epsilon-greedy exploration.

    Parameters
    ----------
    config      : AgentConfig (lr, gamma, epsilon_*, used)
    embeddings  : torch.Tensor of shape (N, D) — pre-trained GNN embeddings
    budget_max  : maximum budget (used to normalise the budget feature)
    device      : torch device string for embedding lookup
    seed        : optional RNG seed
    """

    def __init__(
        self,
        config: AgentConfig,
        embeddings: torch.Tensor,
        budget_max: int,
        device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        self.device = resolve_device(device)
        # SARSA features are computed in numpy — keep embeddings on CPU as np.array
        self.embeddings_np: np.ndarray = embeddings.detach().cpu().numpy().astype(np.float32)
        self.embed_dim: int = self.embeddings_np.shape[1]
        self.budget_max: int = max(1, budget_max)

        # Feature dim: mean_seed_emb (D) + candidate_emb (D) + interaction (D) + budget (1) + bias (1)
        self.feat_dim: int = 3 * self.embed_dim + 2

        # Linear weights — initialised small to start near Q ~ 0
        self._rng = np.random.default_rng(seed)
        self.w: np.ndarray = self._rng.normal(0.0, 0.01, size=self.feat_dim).astype(np.float32)

        # Exploration schedule
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = max(1, config.epsilon_decay)

        # SARSA needs to remember the *previous* (state, action) so the TD
        # update can be performed on the next call when (s', a') become known.
        self._prev_phi: Optional[np.ndarray] = None
        self._prev_q: Optional[float] = None

        # Episode-level metrics
        self._td_errors: List[float] = []

        logger.info(
            "SARSAAgent initialised: feat_dim=%d gamma=%.3f lr=%.4f",
            self.feat_dim, config.gamma, config.learning_rate,
        )

    # ------------------------------------------------------------------ #
    # Feature builder                                                      #
    # ------------------------------------------------------------------ #

    def _phi(self, state: State, action: int) -> np.ndarray:
        """Build the linear feature vector phi(s, a)."""
        if len(state.seed_set) == 0:
            seed_mean = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            idx = np.fromiter(state.seed_set, dtype=np.int64)
            seed_mean = self.embeddings_np[idx].mean(axis=0)

        cand_emb = self.embeddings_np[action]               # (D,)
        interaction = seed_mean * cand_emb                  # Hadamard product (D,)
        budget_norm = np.float32(state.budget / self.budget_max)
        bias = np.float32(1.0)

        return np.concatenate([
            seed_mean,
            cand_emb,
            interaction,
            np.array([budget_norm, bias], dtype=np.float32),
        ], axis=0)

    def _q(self, state: State, action: int) -> float:
        return float(self.w @ self._phi(state, action))

    # ------------------------------------------------------------------ #
    # Exploration schedule                                                 #
    # ------------------------------------------------------------------ #

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self._step / self.epsilon_decay)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-3.0 * progress)

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #

    def _epsilon_greedy(
        self, state: State, valid_actions: List[int],
    ) -> tuple[int, float, np.ndarray]:
        """Pick action; return (action, q_value, phi(s, action))."""
        eps = self.epsilon

        if self._rng.random() < eps:
            action = int(self._rng.choice(valid_actions))
            phi = self._phi(state, action)
            q_val = float(self.w @ phi)
            return action, q_val, phi

        # Greedy over all valid actions — vectorised for speed
        # Build feature matrix Phi (n_valid, feat_dim) and pick argmax w · phi
        if len(state.seed_set) == 0:
            seed_mean = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            idx = np.fromiter(state.seed_set, dtype=np.int64)
            seed_mean = self.embeddings_np[idx].mean(axis=0)

        cand_idx = np.asarray(valid_actions, dtype=np.int64)
        cand_embs = self.embeddings_np[cand_idx]                      # (n, D)

        seed_mean_b = np.broadcast_to(seed_mean, cand_embs.shape)     # (n, D)
        interaction = seed_mean_b * cand_embs                         # (n, D)
        budget_norm = np.float32(state.budget / self.budget_max)
        tail = np.tile(np.array([budget_norm, 1.0], dtype=np.float32), (len(valid_actions), 1))

        Phi = np.concatenate([seed_mean_b, cand_embs, interaction, tail], axis=1)  # (n, F)
        q_values = Phi @ self.w                                       # (n,)

        best_idx = int(np.argmax(q_values))
        action = int(valid_actions[best_idx])
        return action, float(q_values[best_idx]), Phi[best_idx]

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")

        action, q_val, phi = self._epsilon_greedy(state, valid_actions)

        # ------------------ SARSA TD update for the *previous* step ------ #
        # If we have a stored (s_{t-1}, a_{t-1}) and reward r_t, the SARSA
        # target is r_t + gamma * Q(s_t, a_t). We received r_t via update()
        # which stored it as self._pending_reward.
        if self._prev_phi is not None and self._pending_reward is not None:
            target = self._pending_reward + self.config.gamma * q_val
            td_error = target - self._prev_q
            self.w += self.config.learning_rate * td_error * self._prev_phi
            self._td_errors.append(float(td_error))

        # Now this (s, a) becomes the previous step for next time
        self._prev_phi = phi
        self._prev_q = q_val
        self._pending_reward = None

        self.increment_step()
        logger.debug(
            "SARSA selected action=%d Q=%.3f epsilon=%.3f",
            action, q_val, self.epsilon,
        )
        return action

    # ------------------------------------------------------------------ #
    # Learning hook                                                        #
    # ------------------------------------------------------------------ #

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        """
        Cache the reward; the actual SARSA gradient step is applied on the
        *next* call to select_action (so we know a' under current policy).

        For terminal transitions, perform the final update immediately with
        Q-target = r (no bootstrap, since there is no s', a').
        """
        self._pending_reward = float(transition.reward)

        if transition.done and self._prev_phi is not None:
            # Terminal update: target is just r
            target = self._pending_reward
            td_error = target - self._prev_q
            self.w += self.config.learning_rate * td_error * self._prev_phi
            self._td_errors.append(float(td_error))

            metrics = {
                "td_error": float(td_error),
                "epsilon": float(self.epsilon),
                "weight_norm": float(np.linalg.norm(self.w)),
            }

            # Reset for next episode
            self._prev_phi = None
            self._prev_q = None
            self._pending_reward = None
            return metrics

        # Mid-episode: just stash reward; update happens at next select_action
        return None

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset_episode(self) -> None:
        self._prev_phi = None
        self._prev_q = None
        self._pending_reward: Optional[float] = None
        self._td_errors = []

    def end_episode(self, total_reward: float, episode_transitions: List[Transition]) -> None:
        if self._td_errors:
            mean_td = float(np.mean(np.abs(self._td_errors)))
            logger.info(
                "SARSA episode end: total_reward=%.2f mean|TD-error|=%.4f weight_norm=%.3f",
                total_reward, mean_td, float(np.linalg.norm(self.w)),
            )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        super().save(path)
        weights_path = Path(path).with_suffix(".npz")
        np.savez(weights_path, w=self.w)
        logger.info("SARSA weights saved to %s", weights_path)

    def load(self, path: str | Path) -> None:
        super().load(path)
        weights_path = Path(path).with_suffix(".npz")
        if weights_path.exists():
            data = np.load(weights_path)
            self.w = data["w"].astype(np.float32)
            logger.info("SARSA weights loaded from %s", weights_path)
