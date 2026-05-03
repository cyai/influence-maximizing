"""
S2V-DQN: Structure2Vec encoder jointly trained with DQN
(Khalil, Dai, Zhang, Dilkina, Song — NeurIPS 2017,
 "Learning Combinatorial Optimization Algorithms over Graphs").

Key idea
--------
Unlike our `DQNAgent` which uses *pretrained* GNN embeddings as fixed inputs,
S2V-DQN learns its own node representations *end-to-end* with the Q function,
using T iterations of Structure2Vec message passing per forward pass:

    μ_v^0 = 0
    μ_v^{t+1} = ReLU(  W1 · x_v
                     + W2 · Σ_{u ∈ N(v)} μ_u^t
                     + W3 · Σ_{u ∈ N(v)} ReLU(W4 · w_uv) )

where
    x_v   = node tag (1 if v ∈ seed_set, else 0; we extend to scalars in [0,1])
    w_uv  = edge weight (we use the cached IC influence probability P(u → v))

Q-head:
    Q(s, v) = θ1 · ReLU([ θ2 · Σ_u μ_u  ‖  θ3 · μ_v ])

The *graph state* is encoded purely by the seed-set tags x_v and the structure
itself — no external embeddings are needed.  This is the canonical S2V-DQN
formulation; our `DQNAgent`'s mean+max pooling is a downstream simplification.

Training
--------
- Replay buffer of (graph_state, action, reward, next_graph_state, valid_next)
- Standard Bellman target with target net and Polyak updates
- Epsilon-greedy exploration

Implementation notes
--------------------
- We share most plumbing (replay buffer struct, target net, soft updates,
  epsilon schedule) with the `DQNAgent` *philosophy* but reimplement
  everything that touches forward-pass shapes because the encoder is per-node
  rather than per-state-vector.
- Message passing uses `index_add_` scatter — avoids `torch.sparse.mm` which
  is unsupported on Apple MPS (Metal Performance Shaders).
- We use n-step returns (n=1) + Huber loss to match the original paper.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, FrozenSet, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig, resolve_device
from influence_max.environment import State, Transition
from influence_max.graph import GraphData
from influence_max.ic_model import ICDiffusion

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Structure2Vec encoder                                                       #
# --------------------------------------------------------------------------- #


class Structure2Vec(nn.Module):
    """
    Structure2Vec embedding network from Dai et al. 2016, as used in
    Khalil et al. 2017.

    Forward pass
    ------------
    x_v         : (N, F)   node features (seed indicator)
    edge_index  : (2, E)   [src, tgt] edges (symmetrised)
    edge_weight : (E, 1)   IC influence probabilities P(src→tgt)

    All aggregations use `index_add_` scatter — avoids sparse tensor ops
    which are not supported on Apple MPS.
    """

    def __init__(
        self,
        node_feat_dim: int,
        embed_dim: int,
        t_iters: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.t_iters = max(1, int(t_iters))

        self.W1 = nn.Linear(node_feat_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W3 = nn.Linear(embed_dim, embed_dim, bias=False)
        # W4 maps the scalar edge weight to embed_dim
        self.W4 = nn.Linear(1, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,           # (N, F)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight: torch.Tensor, # (E, 1)
    ) -> torch.Tensor:
        """
        Returns
        -------
        mu : (N, embed_dim) node embeddings after T message-passing iterations.
        """
        n = x.size(0)
        src = edge_index[0]   # (E,)
        tgt = edge_index[1]   # (E,)

        # Σ_{u∈N(v)} ReLU(W4 · w_uv)  — edge term, constant across T
        # w_uv = weight on edge (src → tgt), aggregated at tgt
        w_emb = F.relu(self.W4(edge_weight))            # (E, D)
        edge_agg = x.new_zeros(n, self.embed_dim)
        edge_agg.index_add_(0, tgt, w_emb)              # (N, D)
        edge_term = self.W3(edge_agg)                   # (N, D)

        # W1 · x_v — constant across T
        x_term = self.W1(x)                             # (N, D)

        mu = x.new_zeros(n, self.embed_dim)
        for _ in range(self.t_iters):
            # Σ_{u∈N(v)} μ_u  via scatter (src→tgt direction, symmetrised adj)
            neighbor_sum = x.new_zeros(n, self.embed_dim)
            neighbor_sum.index_add_(0, tgt, mu[src])    # (N, D)
            mu = F.relu(x_term + self.W2(neighbor_sum) + edge_term)
        return mu


# --------------------------------------------------------------------------- #
# Q-head                                                                      #
# --------------------------------------------------------------------------- #


class _QHead(nn.Module):
    """
    Q(s, v) = θ1 · ReLU( [ θ2 · Σ_u μ_u  ‖  θ3 · μ_v ] )

    We split θ1 into its two halves θ2 and θ3 implicitly via the concatenation.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.theta2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.theta3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.theta1 = nn.Linear(2 * embed_dim, 1, bias=False)

    def forward(self, mu: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        mu         : (N, D) all node embeddings
        candidates : (M,)   node ids of candidate actions
        returns    : (M,)   Q(s, v) for each candidate
        """
        graph_emb = mu.sum(dim=0, keepdim=True)             # (1, D)
        graph_term = self.theta2(graph_emb)                 # (1, D)
        node_term = self.theta3(mu[candidates])             # (M, D)
        graph_term_b = graph_term.expand(node_term.size(0), -1)
        cat = F.relu(torch.cat([graph_term_b, node_term], dim=-1))   # (M, 2D)
        return self.theta1(cat).squeeze(-1)                  # (M,)


# --------------------------------------------------------------------------- #
# Replay item                                                                 #
# --------------------------------------------------------------------------- #


class _S2VReplayItem:
    __slots__ = ("seed_set", "action", "reward", "next_seed_set", "next_valid", "done")

    def __init__(
        self,
        seed_set: FrozenSet[int],
        action: int,
        reward: float,
        next_seed_set: FrozenSet[int],
        next_valid: Optional[List[int]],
        done: bool,
    ) -> None:
        self.seed_set = seed_set
        self.action = action
        self.reward = reward
        self.next_seed_set = next_seed_set
        self.next_valid = next_valid
        self.done = done


# --------------------------------------------------------------------------- #
# S2V-DQN agent                                                               #
# --------------------------------------------------------------------------- #


class S2VDQNAgent(BaseAgent):
    """
    Structure2Vec encoder jointly trained with a Q-function head, à la
    Khalil et al. 2017.

    Parameters
    ----------
    config        : AgentConfig
    graph         : GraphData (used to build edge_index / edge_weight tensors)
    ic            : ICDiffusion (used only to source edge weights = P(u → v))
    budget_max    : per-episode budget (only used for logging)
    embed_dim     : Structure2Vec embedding dimension
    t_iters       : number of S2V message-passing rounds
    device        : torch device string
    seed          : RNG seed
    """

    def __init__(
        self,
        config: AgentConfig,
        graph: GraphData,
        ic: ICDiffusion,
        budget_max: int,
        embed_dim: int = 64,
        t_iters: int = 4,
        device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        self.device = resolve_device(device)
        self.graph = graph
        self.ic = ic
        self.budget_max = max(1, int(budget_max))
        self.embed_dim = int(embed_dim)
        self.t_iters = max(1, int(t_iters))

        # --- Static graph tensors (built once, reused every forward pass) ---
        n = graph.n_nodes
        self.n_nodes = n

        # Build edge_index + edge_weight (symmetrised, undirected projection).
        # We deliberately avoid sparse tensors — torch.sparse_coo_tensor /
        # torch.sparse.mm are not supported on Apple MPS.
        edge_src: List[int] = []
        edge_tgt: List[int] = []
        edge_weights: List[float] = []
        for (u, v), p in ic.edge_probs.items():
            edge_src.append(u); edge_tgt.append(v); edge_weights.append(p)
            edge_src.append(v); edge_tgt.append(u); edge_weights.append(p)  # symmetrise

        if not edge_src:
            edge_src.append(0); edge_tgt.append(0); edge_weights.append(0.0)

        self.edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long,
                                       device=self.device)
        self.edge_weight = torch.tensor(edge_weights, dtype=torch.float32,
                                        device=self.device).unsqueeze(-1)  # (E, 1)

        logger.info(
            "S2V-DQN graph: n=%d  edges=%d (symmetrised, weights from IC, device=%s)",
            n, len(edge_weights), self.device,
        )

        # --- Networks ---
        node_feat_dim = 1   # seed indicator
        self.encoder = Structure2Vec(node_feat_dim, embed_dim, t_iters).to(self.device)
        self.q_head = _QHead(embed_dim).to(self.device)

        self.target_encoder = Structure2Vec(node_feat_dim, embed_dim, t_iters).to(self.device)
        self.target_q_head = _QHead(embed_dim).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_q_head.load_state_dict(self.q_head.state_dict())
        self.target_encoder.eval()
        self.target_q_head.eval()

        params = list(self.encoder.parameters()) + list(self.q_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

        # --- Replay buffer ---
        self.buffer: Deque[_S2VReplayItem] = deque(maxlen=config.replay_buffer_size)

        # --- Exploration ---
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = max(1, config.epsilon_decay)

        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        self.target_update_freq = max(1, config.target_update_freq)
        self.tau = 0.05

        logger.info(
            "S2V-DQN initialised: embed_dim=%d t_iters=%d device=%s budget=%d",
            embed_dim, t_iters, self.device, budget_max,
        )

    # ------------------------------------------------------------------ #
    # State featurisation                                                  #
    # ------------------------------------------------------------------ #

    def _seed_tags(self, seed_set: FrozenSet[int]) -> torch.Tensor:
        """Build x_v ∈ {0, 1} indicator vector of shape (N, 1)."""
        x = torch.zeros(self.n_nodes, 1, dtype=torch.float32, device=self.device)
        if seed_set:
            idx = torch.tensor(sorted(seed_set), dtype=torch.long, device=self.device)
            x[idx] = 1.0
        return x

    def _encode(self, seed_set: FrozenSet[int], use_target: bool = False) -> torch.Tensor:
        """Run S2V encoder for the given seed set; returns mu of shape (N, D)."""
        x = self._seed_tags(seed_set)
        encoder = self.target_encoder if use_target else self.encoder
        return encoder(x, self.edge_index, self.edge_weight)

    # ------------------------------------------------------------------ #
    # Exploration                                                          #
    # ------------------------------------------------------------------ #

    @property
    def epsilon(self) -> float:
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
            logger.debug("S2V-DQN explore: epsilon=%.3f -> action=%d", eps, action)
            self.increment_step()
            return action

        self.encoder.eval()
        self.q_head.eval()
        with torch.no_grad():
            mu = self._encode(state.seed_set, use_target=False)
            cand = torch.tensor(valid_actions, dtype=torch.long, device=self.device)
            q_values = self.q_head(mu, cand)
        self.encoder.train()
        self.q_head.train()

        best_idx = int(torch.argmax(q_values).item())
        action = int(valid_actions[best_idx])
        logger.debug(
            "S2V-DQN greedy: epsilon=%.3f Q=[%.3f..%.3f] -> action=%d",
            eps, float(q_values.min()), float(q_values.max()), action,
        )
        self.increment_step()
        return action

    # ------------------------------------------------------------------ #
    # Learning                                                             #
    # ------------------------------------------------------------------ #

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        if transition.done:
            next_valid: Optional[List[int]] = None
        else:
            union_activated: FrozenSet[int] = transition.info.get("union_activated", frozenset())
            excluded = set(transition.next_state.seed_set) | set(union_activated)
            nv = [n for n in range(self.n_nodes) if n not in excluded]
            next_valid = nv if nv else None

        item = _S2VReplayItem(
            seed_set=transition.state.seed_set,
            action=int(transition.action),
            reward=float(transition.reward),
            next_seed_set=transition.next_state.seed_set,
            next_valid=next_valid,
            done=bool(transition.done),
        )
        self.buffer.append(item)

        if len(self.buffer) < self.config.batch_size:
            return None

        return self._train_step()

    def _train_step(self) -> Dict[str, float]:
        """
        One DQN gradient step.

        Note: each replay item requires its OWN encoder forward pass because
        the seed-set is a per-item input. We keep the batch small (config.batch_size)
        to keep this tractable. This is faithful to S2V-DQN — there is no easy
        batching across different graph states.
        """
        batch: List[_S2VReplayItem] = random.sample(self.buffer, self.config.batch_size)

        q_pred_list: List[torch.Tensor] = []
        targets: List[float] = []

        for b in batch:
            mu = self._encode(b.seed_set, use_target=False)
            q_pred = self.q_head(
                mu, torch.tensor([b.action], dtype=torch.long, device=self.device)
            )
            q_pred_list.append(q_pred.squeeze())

            with torch.no_grad():
                if b.done or b.next_valid is None:
                    target = b.reward
                else:
                    mu_next = self._encode(b.next_seed_set, use_target=True)
                    cand = torch.tensor(b.next_valid, dtype=torch.long, device=self.device)
                    q_next = self.target_q_head(mu_next, cand)
                    target = b.reward + self.config.gamma * float(q_next.max().item())
                targets.append(target)

        q_pred_tensor = torch.stack(q_pred_list)
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
        loss = F.smooth_l1_loss(q_pred_tensor, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.encoder.parameters()) + list(self.q_head.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        self.optimizer.step()

        if self._step % self.target_update_freq == 0:
            self._soft_update_target()

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_pred_tensor.mean().item()),
            "epsilon": float(self.epsilon),
        }

    def _soft_update_target(self) -> None:
        for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        for tp, p in zip(self.target_q_head.parameters(), self.q_head.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        super().save(path)
        weights_path = Path(path).with_suffix(".pt")
        torch.save({
            "encoder": self.encoder.state_dict(),
            "q_head": self.q_head.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "target_q_head": self.target_q_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, weights_path)
        logger.info("S2V-DQN weights saved to %s", weights_path)

    def load(self, path: str | Path) -> None:
        super().load(path)
        weights_path = Path(path).with_suffix(".pt")
        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.encoder.load_state_dict(ckpt["encoder"])
            self.q_head.load_state_dict(ckpt["q_head"])
            self.target_encoder.load_state_dict(ckpt["target_encoder"])
            self.target_q_head.load_state_dict(ckpt["target_q_head"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("S2V-DQN weights loaded from %s", weights_path)
