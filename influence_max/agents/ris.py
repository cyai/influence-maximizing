"""
Reverse Influence Sampling agent — IMM (Tang, Shi, Xiao 2015).

Algorithm overview
------------------
Influence maximization is reformulated as a *coverage* problem:

  Reverse Reachable (RR) set R_v of a random root node v is the set of nodes
  that *could have* activated v in a single Monte Carlo cascade running
  backward through the directed graph (each edge (u → v) survives with
  probability P(u → v)).

  By symmetry, E[ |Activated(S)| ] = N · P[ S intersects R_v ] for a
  uniformly random root v.  Maximising influence ≈ maximising the fraction
  of RR sets that intersect S → a max-coverage problem.

IMM has two phases:

  1. Sampling — generate θ RR sets, where θ is bounded by a martingale
     concentration argument (Tang et al. 2015 Lemma 6) to ensure that the
     greedy max-coverage solution is a (1 − 1/e − ε)-approximation w.h.p.

  2. Node selection — greedy max-coverage: repeatedly pick the node that
     covers the most as-yet-uncovered RR sets.

Why use IMM here?
-----------------
RIS-family methods are typically 100-1000× faster than CELF, with provable
guarantees and tiny memory.  Our IMM uses the *embedding-based* edge
probabilities P(u→v) = σ(H[u]·H[v]) so it is directly comparable to the other
agents (no probability-model confound).

Implementation notes
--------------------
- IMM's full sampling lower bound (LB-Phase + Sampling-Phase) is intricate
  and depends on the OPT estimate.  We use the simpler "fixed-θ" variant:
    θ = (8 + 2ε) · N · ( ln(2/δ) + ln(C(N, k)) ) / (OPT_LB · ε²)
  with OPT_LB approximated via degree-weighted heuristic.  Practical θ is
  capped at θ_max for runtime safety.
- The agent precomputes the full seed set on the first select_action() call
  and then yields nodes one at a time on subsequent calls (so it fits the
  per-step BaseAgent interface).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig
from influence_max.environment import State, Transition
from influence_max.graph import GraphData
from influence_max.ic_model import ICDiffusion

logger = logging.getLogger(__name__)


class IMMAgent(BaseAgent):
    """
    Influence Maximization via Martingales (Tang et al. 2015).

    Parameters
    ----------
    config            : AgentConfig
    graph             : GraphData
    ic                : ICDiffusion (used only to fetch edge_probs cache)
    budget            : per-episode seed budget k
    epsilon           : approximation error parameter (smaller → more RR sets)
    delta             : failure probability (smaller → more RR sets)
    theta_max         : cap on the number of RR sets (runtime safety)
    seed              : RNG seed
    """

    def __init__(
        self,
        config: AgentConfig,
        graph: GraphData,
        ic: ICDiffusion,
        budget: int,
        epsilon: float = 0.1,
        delta: Optional[float] = None,
        theta_max: int = 200_000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        self.graph = graph
        self.ic = ic
        self.budget = max(1, budget)
        self.epsilon = float(epsilon)
        self.delta = float(delta) if delta is not None else 1.0 / max(1, graph.n_nodes)
        self.theta_max = int(theta_max)
        self._rng = np.random.default_rng(seed)

        # Pre-built reverse adjacency for fast reverse BFS:
        #   in_neighbors[v] = list of u such that (u, v) ∈ E
        # ic.edge_probs is a dict (u, v) -> p
        self._in_neighbors: Dict[int, List[int]] = graph.in_neighbors

        # Precomputed seed set + iterator over it
        self._planned_seeds: List[int] = []
        self._seed_iter_idx: int = 0
        self._n_rr_sets: int = 0

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset_episode(self) -> None:
        self._planned_seeds = []
        self._seed_iter_idx = 0
        self._n_rr_sets = 0

    def end_episode(
        self, total_reward: float, episode_transitions: List[Transition]
    ) -> None:
        logger.info(
            "IMM episode done: %d RR sets, planned seeds: %s",
            self._n_rr_sets,
            self._planned_seeds,
        )

    # ------------------------------------------------------------------ #
    # RR-set sampling                                                      #
    # ------------------------------------------------------------------ #

    def _sample_rr_set(self, root: int) -> Set[int]:
        """
        One reverse BFS from `root`. Each in-edge (u → root) is included with
        prob P(u → root). The RR set is the closure of all nodes reachable
        backward via included edges.
        """
        edge_probs = self.ic.edge_probs
        rr: Set[int] = {root}
        frontier: List[int] = [root]

        while frontier:
            new_frontier: List[int] = []
            for v in frontier:
                for u in self._in_neighbors.get(v, []):
                    if u in rr:
                        continue
                    p = edge_probs.get((u, v), 0.0)
                    if p > 0.0 and self._rng.random() < p:
                        rr.add(u)
                        new_frontier.append(u)
            frontier = new_frontier
        return rr

    def _compute_theta(self) -> int:
        """
        Number of RR sets to sample.

        Standard IMM estimate (simplified, fixed-OPT-lower-bound version):
            theta = (8 + 2ε) · N · ( ln(2/δ) + ln(C(N, k)) ) / (OPT_LB · ε²)

        We approximate OPT_LB ≈ k (a very loose lower bound: at minimum the
        seed set covers itself).  This intentionally over-samples relative
        to the tight IMM bound but is safer and still tractable.
        """
        n = self.graph.n_nodes
        k = self.budget
        eps = self.epsilon
        # log(C(n, k)) using the Stirling-style logfactorial
        log_nck = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        log_term = math.log(2.0 / self.delta) + log_nck
        opt_lb = float(k)
        theta = (8.0 + 2.0 * eps) * n * log_term / (opt_lb * eps * eps)
        theta_int = max(int(math.ceil(theta)), 1000)
        return min(theta_int, self.theta_max)

    def _sample_all_rr_sets(self, theta: int) -> List[Set[int]]:
        """Sample θ RR sets from uniformly random roots."""
        n = self.graph.n_nodes
        rng = self._rng
        rr_sets: List[Set[int]] = []
        log_every = max(1, theta // 10)
        for i in range(theta):
            root = int(rng.integers(0, n))
            rr_sets.append(self._sample_rr_set(root))
            if (i + 1) % log_every == 0:
                logger.debug("IMM sampling: %d / %d RR sets...", i + 1, theta)
        return rr_sets

    # ------------------------------------------------------------------ #
    # Greedy max-coverage over RR sets                                     #
    # ------------------------------------------------------------------ #

    def _select_seeds_max_cover(
        self,
        rr_sets: List[Set[int]],
        valid_nodes: Set[int],
    ) -> List[int]:
        """
        Greedy max-coverage: at each step pick the node that appears in the
        most as-yet-uncovered RR sets.

        Build inverted index: node -> set of RR-set indices that contain it.
        """
        node_to_rr: Dict[int, Set[int]] = defaultdict(set)
        for idx, rr in enumerate(rr_sets):
            for v in rr:
                if v in valid_nodes:
                    node_to_rr[v].add(idx)

        seeds: List[int] = []
        covered: Set[int] = set()
        for _ in range(self.budget):
            if not node_to_rr:
                break
            # argmax over coverage of *uncovered* RR sets
            best_node = -1
            best_cov = -1
            for v, indices in node_to_rr.items():
                cov = len(indices - covered)
                if cov > best_cov:
                    best_cov = cov
                    best_node = v
            if best_node == -1 or best_cov == 0:
                # All remaining RR sets are covered; fall back to degree
                logger.debug("IMM: greedy converged early after %d seeds", len(seeds))
                break
            seeds.append(best_node)
            covered |= node_to_rr.pop(best_node)

        return seeds

    # ------------------------------------------------------------------ #
    # Plan all seeds on first call                                         #
    # ------------------------------------------------------------------ #

    def _plan(self, valid_actions: List[int]) -> None:
        theta = self._compute_theta()
        logger.info(
            "IMM: sampling %d RR sets (eps=%.3f, delta=%.2e)...",
            theta,
            self.epsilon,
            self.delta,
        )

        rr_sets = self._sample_all_rr_sets(theta)
        self._n_rr_sets = len(rr_sets)

        valid_set = set(valid_actions)
        seeds = self._select_seeds_max_cover(rr_sets, valid_set)
        self._planned_seeds = seeds

        logger.info("IMM: planned seeds (top %d) = %s", len(seeds), seeds)

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

        # Plan once on first call
        if (
            not self._planned_seeds
            and self._seed_iter_idx == 0
            and self._n_rr_sets == 0
        ):
            self._plan(valid_actions)

        valid_set = set(valid_actions)
        # Skip planned seeds that are no longer valid (already activated etc.)
        while self._seed_iter_idx < len(self._planned_seeds):
            v = self._planned_seeds[self._seed_iter_idx]
            self._seed_iter_idx += 1
            if v in valid_set:
                self.increment_step()
                logger.debug("IMM yielding planned seed=%d", v)
                return v

        # Plan exhausted (all planned seeds invalidated) — fall back to highest
        # out-degree among remaining valid candidates
        fallback = max(valid_actions, key=lambda u: self.graph.out_degree[u])
        self.increment_step()
        logger.warning(
            "IMM plan exhausted, falling back to degree heuristic: %d", fallback
        )
        return fallback

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        return None
