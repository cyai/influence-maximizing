"""
LinUCB contextual bandit agent for influence maximization.

Problem framing
---------------
The bandit reformulation discards the sequential / Markovian structure: each
seed selection is treated as an independent contextual decision.

  - Arm           = candidate node v
  - Context x_v   = features of (current_state, v)
  - Reward        = marginal influence gain observed after picking v
  - Goal          = maximise cumulative reward across all selections

This is intentionally simpler than DQN/SARSA and serves as an instructive
contrast: it shows what is *lost* when we ignore long-horizon planning.

Why LinUCB (Li et al., 2010)?
- Provably efficient regret bound O(sqrt(T)).
- Closed-form update — no SGD, no replay buffer, no neural network.
- Naturally exploration-aware via the upper-confidence-bound term — no
  epsilon-greedy schedule needed.

Algorithm (single shared theta across arms)
-------------------------------------------
Maintain:
    A  : (D x D)  symmetric positive-definite matrix, init = lambda * I
    b  : (D,)     vector,                              init = 0
    A_inv : cached inverse, updated via Sherman-Morrison on each rank-1 update

For each candidate arm a with context x_a:
    theta_hat = A_inv @ b
    p_a       = theta_hat . x_a + alpha * sqrt(x_a^T A_inv x_a)
Pick a* = argmax_a p_a.

After observing reward r:
    A     <- A + x_a* x_a*^T          (use Sherman-Morrison on A_inv)
    b     <- b + r * x_a*

Feature design
--------------
x_v = [ H[v],                               # candidate embedding (D)
        mean(H[seed_set]) * H[v],           # state-action interaction (D)
        budget_remaining / budget_max,      # 1
        1.0 ]                               # bias
Total feature dim = 2*D + 2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig
from influence_max.environment import State, Transition

logger = logging.getLogger(__name__)


class LinUCBAgent(BaseAgent):
    """
    Contextual linear bandit with upper-confidence-bound exploration.

    Parameters
    ----------
    config       : AgentConfig (uses learning_rate as the regularisation
                   strength `lambda` and `epsilon_start` as the alpha exploration
                   parameter for clean reuse — no new config field needed)
    embeddings   : torch.Tensor of shape (N, D) — pre-trained GNN embeddings
    budget_max   : maximum budget (used to normalise the budget feature)
    alpha        : UCB exploration coefficient (overrides config if provided)
    seed         : optional RNG seed (only used for tie-breaking)
    """

    def __init__(
        self,
        config: AgentConfig,
        embeddings: torch.Tensor,
        budget_max: int,
        alpha: float = 1.0,
        ridge_lambda: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        # L2-normalise embeddings: standard LinUCB regret bounds assume ||x|| <= 1.
        # Without this, rewards in the hundreds combined with unbounded GAE
        # embeddings would cause A_inv @ b to overflow.
        emb_np = embeddings.detach().cpu().numpy().astype(np.float64)
        norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.embeddings_np: np.ndarray = emb_np / norms

        self.embed_dim: int = self.embeddings_np.shape[1]
        self.budget_max: int = max(1, budget_max)
        # Reward scale — divides observed rewards before LinUCB update so that
        # b stays in a numerically reasonable range. Computed adaptively from
        # the running max absolute reward seen so far.
        self._reward_scale: float = 1.0

        # Feature dim: candidate (D) + interaction (D) + budget (1) + bias (1)
        self.feat_dim: int = 2 * self.embed_dim + 2

        # Ridge regression initial matrices
        self.A: np.ndarray = ridge_lambda * np.eye(self.feat_dim, dtype=np.float64)
        self.A_inv: np.ndarray = (1.0 / ridge_lambda) * np.eye(self.feat_dim, dtype=np.float64)
        self.b: np.ndarray = np.zeros(self.feat_dim, dtype=np.float64)

        self.alpha: float = float(alpha)

        self._rng = np.random.default_rng(seed)

        # Track context of last action for the update step
        self._last_context: Optional[np.ndarray] = None

        logger.info(
            "LinUCBAgent initialised: feat_dim=%d alpha=%.3f lambda=%.3f",
            self.feat_dim, self.alpha, ridge_lambda,
        )

    # ------------------------------------------------------------------ #
    # Feature builder                                                      #
    # ------------------------------------------------------------------ #

    def _contexts(self, state: State, valid_actions: List[int]) -> np.ndarray:
        """Build context matrix X of shape (n_valid, feat_dim) for all candidates."""
        if len(state.seed_set) == 0:
            seed_mean = np.zeros(self.embed_dim, dtype=np.float64)
        else:
            idx = np.fromiter(state.seed_set, dtype=np.int64)
            seed_mean = self.embeddings_np[idx].mean(axis=0)

        cand_idx = np.asarray(valid_actions, dtype=np.int64)
        cand_embs = self.embeddings_np[cand_idx]                     # (n, D)
        seed_mean_b = np.broadcast_to(seed_mean, cand_embs.shape)    # (n, D)
        interaction = seed_mean_b * cand_embs                        # (n, D)

        budget_norm = state.budget / self.budget_max
        tail = np.tile(np.array([budget_norm, 1.0], dtype=np.float64), (len(valid_actions), 1))

        X = np.concatenate([cand_embs, interaction, tail], axis=1)   # (n, F)
        # L2-normalise each row so all contexts satisfy ||x|| <= 1
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-12)
        return X / row_norms

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

        X = self._contexts(state, valid_actions)                      # (n, F)

        # NOTE: Apple Accelerate BLAS triggers spurious "divide by zero" warnings
        # in matmul when an operand is all-zero (e.g. b = 0 initially). The
        # results are correct; we just suppress the noise.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            theta = self.A_inv @ self.b                                # (F,)
            mean_est = X @ theta                                       # (n,)
            AinvX = X @ self.A_inv                                     # (n, F)

        widths = np.sqrt(np.maximum(np.einsum("ij,ij->i", AinvX, X), 1e-12))
        ucb = mean_est + self.alpha * widths                          # (n,)

        # Argmax with random tie-break
        max_val = ucb.max()
        # Tolerate floating-point ties
        candidates = np.flatnonzero(ucb >= max_val - 1e-9)
        if len(candidates) == 1:
            best_idx = int(candidates[0])
        else:
            best_idx = int(self._rng.choice(candidates))

        action = int(valid_actions[best_idx])
        self._last_context = X[best_idx].copy()                       # cache for update()

        self.increment_step()
        logger.debug(
            "LinUCB selected action=%d ucb=%.3f mean=%.3f width=%.3f",
            action, ucb[best_idx], mean_est[best_idx], widths[best_idx],
        )
        return action

    # ------------------------------------------------------------------ #
    # Learning                                                             #
    # ------------------------------------------------------------------ #

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        """Apply rank-1 updates to A and b given the observed reward."""
        if self._last_context is None:
            return None

        x = self._last_context
        r_raw = float(transition.reward)

        # Adaptive reward scaling: keep |r_scaled| roughly in [0, 1] for
        # numerical stability of LinUCB's closed-form linear regression.
        self._reward_scale = max(self._reward_scale, abs(r_raw))
        r = r_raw / self._reward_scale

        # b update
        self.b += r * x

        # A update (rank-1): A <- A + x x^T
        self.A += np.outer(x, x)

        # Sherman-Morrison rank-1 inverse update:
        # (A + x x^T)^{-1} = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            Ainv_x = self.A_inv @ x
        denom = 1.0 + float(x @ Ainv_x)
        if denom > 1e-12:
            self.A_inv -= np.outer(Ainv_x, Ainv_x) / denom
        # else: skip update — should be impossible since A is PSD

        self._last_context = None

        return {
            "reward_raw": r_raw,
            "reward_scaled": r,
            "reward_scale": self._reward_scale,
            "trace_A_inv": float(np.trace(self.A_inv)),
            "norm_b": float(np.linalg.norm(self.b)),
        }

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset_episode(self) -> None:
        # Bandit state persists across episodes (cumulative learning),
        # but per-step caches are reset.
        self._last_context = None

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        super().save(path)
        weights_path = Path(path).with_suffix(".npz")
        np.savez(weights_path, A=self.A, A_inv=self.A_inv, b=self.b,
                 alpha=self.alpha, reward_scale=self._reward_scale)
        logger.info("LinUCB state saved to %s", weights_path)

    def load(self, path: str | Path) -> None:
        super().load(path)
        weights_path = Path(path).with_suffix(".npz")
        if weights_path.exists():
            data = np.load(weights_path)
            self.A = data["A"]
            self.A_inv = data["A_inv"]
            self.b = data["b"]
            self.alpha = float(data["alpha"])
            if "reward_scale" in data.files:
                self._reward_scale = float(data["reward_scale"])
            logger.info("LinUCB state loaded from %s", weights_path)
