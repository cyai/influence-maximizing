"""
Community-based influence maximization agent — Louvain partition + per-community greedy.

Algorithm
---------
1. Detect communities on the *undirected* projection of the directed graph
   using Louvain modularity optimisation (Blondel et al. 2008,
   `networkx.community.louvain_communities`).
2. Allocate the seed budget proportionally to community size:
       budget_c = round(k · |C| / N)
   with one round of leftover redistribution to ensure Σ budget_c = k.
3. For each community c with budget_c > 0, run greedy-MC inside the
   subgraph induced by c.

Strengths (per project plan):
  - Reduces candidate space from N to ≤ N (per-community max)
  - Encourages spread across structurally distinct regions

Limitations (per project plan):
  - Quality depends on partition quality
  - Inter-community spread is poorly captured (greedy never "looks across"
    community boundaries to combine bridge nodes)

Implementation
--------------
The agent precomputes the seed plan once on the first `select_action()` call
(after the partition is built) and then yields planned seeds one at a time on
subsequent calls.  This makes it compatible with the per-step BaseAgent
interface used by the env / runner.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig
from influence_max.environment import State, Transition
from influence_max.graph import GraphData
from influence_max.ic_model import MonteCarloEstimator

logger = logging.getLogger(__name__)


class LouvainAgent(BaseAgent):
    """
    Louvain community partition + per-community greedy MC.

    Parameters
    ----------
    config       : AgentConfig (mc_rollouts controls per-candidate MC count)
    mc           : MonteCarloEstimator (shared with env, used for spread eval)
    graph        : GraphData
    budget       : per-episode seed budget k (needed for budget allocation)
    resolution   : Louvain resolution (>1 = more, smaller communities)
    seed         : RNG seed for Louvain (it has a stochastic step)
    """

    def __init__(
        self,
        config: AgentConfig,
        mc: MonteCarloEstimator,
        graph: GraphData,
        budget: int,
        resolution: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        self.mc = mc
        self.graph = graph
        self.budget = max(1, budget)
        self.resolution = float(resolution)
        self._seed = seed

        # Set during _plan(); reset per-episode
        self._planned_seeds: List[int] = []
        self._seed_iter_idx: int = 0
        self._communities: Optional[List[Set[int]]] = None

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset_episode(self) -> None:
        # Re-plan from scratch each episode (env state may have shifted)
        self._planned_seeds = []
        self._seed_iter_idx = 0

    def end_episode(
        self, total_reward: float, episode_transitions: List[Transition]
    ) -> None:
        logger.info(
            "Louvain episode done: %d communities, planned seeds=%s",
            len(self._communities) if self._communities else 0,
            self._planned_seeds,
        )

    # ------------------------------------------------------------------ #
    # Partition + plan                                                     #
    # ------------------------------------------------------------------ #

    def _detect_communities(self) -> List[Set[int]]:
        """Run Louvain on the undirected projection of the directed graph."""
        if self._communities is not None:
            return self._communities

        logger.info(
            "Louvain: building undirected projection (n=%d)...", self.graph.n_nodes
        )
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.node_ids)
        g.add_edges_from(self.graph.edges)
        u = g.to_undirected()
        u.remove_edges_from(nx.selfloop_edges(u))

        logger.info(
            "Louvain: detecting communities (resolution=%.2f)...", self.resolution
        )
        communities = nx.community.louvain_communities(
            u,
            resolution=self.resolution,
            seed=self._seed,
        )
        # Sort by size descending so the largest are processed first
        communities = sorted([set(c) for c in communities], key=len, reverse=True)
        logger.info(
            "Louvain: found %d communities, sizes (top 10): %s",
            len(communities),
            [len(c) for c in communities[:10]],
        )
        self._communities = communities
        return communities

    def _allocate_budget(self, communities: List[Set[int]]) -> List[int]:
        """
        Allocate `self.budget` seeds across communities proportionally to size.

        Uses the largest-remainder method to ensure Σ budget_c = budget.
        """
        n = self.graph.n_nodes
        k = self.budget
        sizes = np.array([len(c) for c in communities], dtype=np.float64)
        raw = sizes * (k / n)
        floor = np.floor(raw).astype(int)
        remainder = raw - floor
        leftover = k - int(floor.sum())

        # Distribute leftover seeds to communities with highest remainder
        if leftover > 0:
            order = np.argsort(remainder)[::-1]
            for idx in order[:leftover]:
                floor[idx] += 1

        # Cap each per-community budget at community size
        for i, c in enumerate(communities):
            floor[i] = min(int(floor[i]), len(c))

        # If we lost seeds due to capping, redistribute to under-allocated communities
        deficit = k - int(floor.sum())
        if deficit > 0:
            for i, c in enumerate(communities):
                if deficit == 0:
                    break
                slack = len(c) - int(floor[i])
                if slack > 0:
                    add = min(slack, deficit)
                    floor[i] += add
                    deficit -= add

        budget_per_community = floor.tolist()
        logger.info(
            "Louvain: budget allocation (top 10): %s", budget_per_community[:10]
        )
        return budget_per_community

    def _greedy_within_community(
        self,
        community: Set[int],
        b: int,
        rollouts: int,
    ) -> List[int]:
        """
        Run greedy-MC inside the subgraph induced by `community`.

        Each candidate's marginal gain is estimated against the *full* graph's
        IC model (we only restrict the candidate set to the community — spread
        itself is computed globally because the IC cascade isn't bounded by
        community walls).
        """
        if b <= 0 or not community:
            return []

        chosen: List[int] = []
        seed_set = frozenset()
        base_spread = 0.0

        for _ in range(b):
            best_node = -1
            best_gain = -np.inf
            for v in community:
                if v in seed_set:
                    continue
                new_seeds = seed_set | frozenset([v])
                gain = (
                    self.mc.estimate(new_seeds, n_simulations=rollouts).mean_spread
                    - base_spread
                )
                if gain > best_gain:
                    best_gain = gain
                    best_node = v
            if best_node == -1:
                break
            chosen.append(best_node)
            seed_set = seed_set | frozenset([best_node])
            base_spread = base_spread + best_gain

        return chosen

    def _plan(self) -> None:
        """Build the seed plan: detect communities → allocate → per-comm greedy."""
        communities = self._detect_communities()
        budgets = self._allocate_budget(communities)
        rollouts = self.config.mc_rollouts

        all_seeds: List[int] = []
        for c, b in zip(communities, budgets):
            if b == 0:
                continue
            logger.debug(
                "Louvain: greedy-MC inside community size=%d budget=%d", len(c), b
            )
            picks = self._greedy_within_community(c, b, rollouts)
            all_seeds.extend(picks)
            if len(all_seeds) >= self.budget:
                break

        self._planned_seeds = all_seeds[: self.budget]
        logger.info("Louvain: final planned seeds = %s", self._planned_seeds)

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
        if not self._planned_seeds:
            self._plan()

        valid_set = set(valid_actions)
        while self._seed_iter_idx < len(self._planned_seeds):
            v = self._planned_seeds[self._seed_iter_idx]
            self._seed_iter_idx += 1
            if v in valid_set:
                self.increment_step()
                logger.debug("Louvain yielding planned seed=%d", v)
                return v

        # Fallback: highest out-degree among remaining valid candidates
        fallback = max(valid_actions, key=lambda u: self.graph.out_degree[u])
        self.increment_step()
        logger.warning(
            "Louvain plan exhausted, falling back to degree heuristic: %d", fallback
        )
        return fallback

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        return None
