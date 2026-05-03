"""
Deep Q-Network agent for influence maximization.

Architecture
------------
State representation s_t:
    Order-invariant aggregation of seed-set embeddings + budget signal:
        s_t = [ mean(H[seed_set]),  max(H[seed_set]),  budget_remaining / budget ]
    For an empty seed set, the aggregated parts are zero vectors.

Action representation a:
    The candidate node's embedding H[a].

Q-network:
    MLP that outputs Q(s, a) = f_theta([s_t ‖ H[a]]).
    This is the standard "action-conditional" formulation for large discrete
    action spaces — we evaluate Q for each candidate independently rather
    than producing a 6301-way vector. It generalises across nodes and is
    parameter-efficient.

Training:
    - Replay buffer holds (s, a, r, s', valid_actions_next, done) tuples.
    - Target network with soft Polyak updates: theta_target <- tau*theta + (1-tau)*theta_target.
    - Loss: smooth L1 (Huber) on Q(s, a) vs. r + gamma * max_{a'} Q_target(s', a').
    - Epsilon-greedy exploration with exponential decay over global agent steps.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig, resolve_device
from influence_max.environment import State, Transition

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Replay buffer transition                                                    #
# --------------------------------------------------------------------------- #


class _ReplayItem:
    """A single transition stored in the replay buffer.

    We keep `valid_actions_next` so the Bellman target can correctly compute
    max over valid (non-seed, non-activated) actions at the next state.
    """

    __slots__ = ("state_feat", "action_emb", "reward", "next_state_feat",
                 "next_valid_embs", "done")

    def __init__(
        self,
        state_feat: np.ndarray,
        action_emb: np.ndarray,
        reward: float,
        next_state_feat: np.ndarray,
        next_valid_embs: Optional[np.ndarray],
        done: bool,
    ) -> None:
        self.state_feat = state_feat
        self.action_emb = action_emb
        self.reward = reward
        self.next_state_feat = next_state_feat
        self.next_valid_embs = next_valid_embs   # (n_valid, D) or None if done
        self.done = done


# --------------------------------------------------------------------------- #
# Q-Network                                                                   #
# --------------------------------------------------------------------------- #


class QNetwork(nn.Module):
    """MLP Q-function: Q(s, a) where input is [state_feat ‖ action_emb]."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        in_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_feat: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for batched state/action pairs.

        Shapes
        ------
        state_feat : (B, state_dim)  OR  (state_dim,) — broadcast to match action_emb
        action_emb : (B, action_dim)
        returns    : (B,)  scalar Q-values
        """
        if state_feat.dim() == 1:
            state_feat = state_feat.unsqueeze(0).expand(action_emb.size(0), -1)
        x = torch.cat([state_feat, action_emb], dim=-1)
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# DQN Agent                                                                   #
# --------------------------------------------------------------------------- #


class DQNAgent(BaseAgent):
    """
    Q-learning agent with neural-network function approximation.

    Parameters
    ----------
    config      : AgentConfig (lr, gamma, epsilon_*, replay_buffer_size,
                  batch_size, target_update_freq used)
    embeddings  : torch.Tensor of shape (N, D) — pre-trained GNN embeddings
    budget_max  : maximum budget (used to normalise the budget feature)
    device      : torch device string ("auto" | "mps" | "cuda" | "cpu")
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
        self.embeddings: torch.Tensor = embeddings.float().to(self.device)
        self.embed_dim: int = self.embeddings.size(1)
        self.budget_max: int = max(1, budget_max)

        # State feature = [mean_pool(D), max_pool(D), budget_norm(1)]
        self.state_dim: int = 2 * self.embed_dim + 1
        self.action_dim: int = self.embed_dim

        # --- Networks ---
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.learning_rate)

        # --- Replay buffer ---
        self.buffer: Deque[_ReplayItem] = deque(maxlen=config.replay_buffer_size)

        # --- Exploration schedule ---
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = max(1, config.epsilon_decay)

        # --- RNG ---
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        # Target net update frequency (in agent steps)
        self.target_update_freq = max(1, config.target_update_freq)
        # Soft update coefficient (Polyak averaging) — used at every target sync
        self.tau = 0.05

        logger.info(
            "DQNAgent initialised: state_dim=%d action_dim=%d device=%s",
            self.state_dim, self.action_dim, self.device,
        )

    # ------------------------------------------------------------------ #
    # Feature builders                                                     #
    # ------------------------------------------------------------------ #

    def _state_features(self, state: State) -> np.ndarray:
        """Build a fixed-size state feature vector from the seed set + budget."""
        if len(state.seed_set) == 0:
            seed_mean = np.zeros(self.embed_dim, dtype=np.float32)
            seed_max = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            idx = torch.tensor(sorted(state.seed_set), dtype=torch.long, device=self.device)
            seed_embs = self.embeddings[idx]              # (k, D)
            seed_mean = seed_embs.mean(dim=0).cpu().numpy().astype(np.float32)
            seed_max = seed_embs.max(dim=0).values.cpu().numpy().astype(np.float32)

        budget_norm = np.array([state.budget / self.budget_max], dtype=np.float32)
        return np.concatenate([seed_mean, seed_max, budget_norm], axis=0)

    def _action_embeddings(self, actions: List[int]) -> np.ndarray:
        """Lookup the embedding rows for a list of node ids."""
        idx = torch.tensor(actions, dtype=torch.long, device=self.device)
        return self.embeddings[idx].cpu().numpy().astype(np.float32)  # (n, D)

    # ------------------------------------------------------------------ #
    # Exploration                                                          #
    # ------------------------------------------------------------------ #

    @property
    def epsilon(self) -> float:
        """Exponential decay from epsilon_start to epsilon_end over `epsilon_decay` steps."""
        progress = min(1.0, self._step / self.epsilon_decay)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-3.0 * progress)

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")

        eps = self.epsilon
        if self._rng.random() < eps:
            action = int(self._rng.choice(valid_actions))
            logger.debug("DQN explore: epsilon=%.3f -> random action=%d", eps, action)
            self.increment_step()
            return action

        # Greedy: argmax_a Q(s, a) over valid candidates only
        state_feat = torch.from_numpy(self._state_features(state)).to(self.device)
        action_embs = torch.from_numpy(self._action_embeddings(valid_actions)).to(self.device)

        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_feat, action_embs)   # (n_valid,)
        self.q_net.train()

        best_idx = int(torch.argmax(q_values).item())
        action = int(valid_actions[best_idx])
        logger.debug(
            "DQN greedy: epsilon=%.3f Q=[%.3f..%.3f] -> action=%d",
            eps, float(q_values.min()), float(q_values.max()), action,
        )
        self.increment_step()
        return action

    # ------------------------------------------------------------------ #
    # Learning                                                             #
    # ------------------------------------------------------------------ #

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        """Push transition into replay buffer and run one SGD step on a sampled batch."""
        # Build features for the transition
        s_feat = self._state_features(transition.state)
        a_emb = self._action_embeddings([transition.action])[0]    # (D,)
        s_next_feat = self._state_features(transition.next_state)

        # Build candidate embeddings for the next state's valid actions.
        # We use info["union_activated"] to exclude already-activated nodes.
        if transition.done:
            next_valid_embs = None
        else:
            union_activated: FrozenSet[int] = transition.info.get("union_activated", frozenset())
            n_total = self.embeddings.size(0)
            excluded = set(transition.next_state.seed_set) | set(union_activated)
            next_valid = [n for n in range(n_total) if n not in excluded]
            if not next_valid:
                next_valid_embs = None
            else:
                next_valid_embs = self._action_embeddings(next_valid)

        item = _ReplayItem(
            state_feat=s_feat,
            action_emb=a_emb,
            reward=float(transition.reward),
            next_state_feat=s_next_feat,
            next_valid_embs=next_valid_embs,
            done=bool(transition.done),
        )
        self.buffer.append(item)

        # Train when we have enough samples
        if len(self.buffer) < self.config.batch_size:
            return None

        return self._train_step()

    def _train_step(self) -> Dict[str, float]:
        """Sample a batch and perform one DQN gradient update."""
        batch: List[_ReplayItem] = random.sample(self.buffer, self.config.batch_size)

        # Stack current-state features / actions / rewards
        s_feats = torch.from_numpy(
            np.stack([b.state_feat for b in batch], axis=0)
        ).to(self.device)
        a_embs = torch.from_numpy(
            np.stack([b.action_emb for b in batch], axis=0)
        ).to(self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        # Q(s, a) — predicted
        q_pred = self.q_net(s_feats, a_embs)   # (B,)

        # Compute target: r + gamma * max_{a'} Q_target(s', a') (for non-terminal)
        with torch.no_grad():
            q_next_max = torch.zeros(self.config.batch_size, device=self.device)
            for i, b in enumerate(batch):
                if b.done or b.next_valid_embs is None or len(b.next_valid_embs) == 0:
                    continue
                ns_feat = torch.from_numpy(b.next_state_feat).to(self.device)
                na_embs = torch.from_numpy(b.next_valid_embs).to(self.device)
                q_next = self.target_net(ns_feat, na_embs)
                q_next_max[i] = q_next.max()

            target = rewards + self.config.gamma * q_next_max * (1.0 - dones)

        loss = F.smooth_l1_loss(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodic soft target update (Polyak averaging)
        if self._step % self.target_update_freq == 0:
            self._soft_update_target()

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_pred.mean().item()),
            "epsilon": float(self.epsilon),
        }

    def _soft_update_target(self) -> None:
        for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        super().save(path)
        weights_path = Path(path).with_suffix(".pt")
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, weights_path)
        logger.info("DQN weights saved to %s", weights_path)

    def load(self, path: str | Path) -> None:
        super().load(path)
        weights_path = Path(path).with_suffix(".pt")
        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.q_net.load_state_dict(ckpt["q_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("DQN weights loaded from %s", weights_path)
