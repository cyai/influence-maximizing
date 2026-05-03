"""
Centrality and heuristic-based influence maximization agents.

All four agents share the same pattern:

  1. On the first call, build a NetworkX DiGraph (or its undirected projection
     where required) and precompute a node ranking score.
  2. Each `select_action` returns the highest-scoring valid candidate.

Algorithms
----------
PageRankAgent
    Score(v) = PageRank(v) on the directed graph (edges represent endorsement).

KShellAgent
    Score(v) = core number from k-core decomposition. NetworkX implements this
    on the *undirected* projection — k-shell is canonically an undirected
    measure.  Ties broken by out-degree.

BetweennessAgent
    Score(v) = betweenness centrality (fraction of shortest paths through v).
    O(N·M) — slow on large graphs; we run it once per episode and cache.

DegreeDiscountAgent (Chen, Wang, Yang 2009)
    State-aware: at each step, discount the candidate score by the number of
    neighbours already seeded:
        score(v) = d_v − 2·t_v − (d_v − t_v)·t_v·p
    where t_v = #seed-set neighbours of v, p = mean influence prob.
    This crudely accounts for the diminishing marginal return when neighbours
    of seeds are picked.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig
from influence_max.environment import State, Transition
from influence_max.graph import GraphData
from influence_max.ic_model import ICDiffusion

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helper                                                             #
# --------------------------------------------------------------------------- #


def _build_nx_digraph(graph: GraphData) -> nx.DiGraph:
    """Construct a NetworkX DiGraph from our lightweight GraphData."""
    g = nx.DiGraph()
    g.add_nodes_from(graph.node_ids)
    g.add_edges_from(graph.edges)
    return g


# --------------------------------------------------------------------------- #
# PageRank                                                                     #
# --------------------------------------------------------------------------- #


class PageRankAgent(BaseAgent):
    """
    Top-k by PageRank on the directed graph.

    PageRank is interpreted as long-run probability mass under a random surfer
    on the network — high-PR nodes are structurally well-connected sinks, often
    correlated with influence under IC.
    """

    def __init__(
        self,
        config: AgentConfig,
        graph: GraphData,
        alpha: float = 0.85,
    ) -> None:
        super().__init__(config)
        self.graph = graph
        self.alpha = float(alpha)
        self._scores: Optional[np.ndarray] = None

    def _ensure_scores(self) -> None:
        if self._scores is not None:
            return
        logger.info("PageRank: computing scores (alpha=%.2f, n=%d)...",
                    self.alpha, self.graph.n_nodes)
        g = _build_nx_digraph(self.graph)
        pr = nx.pagerank(g, alpha=self.alpha, max_iter=200, tol=1e-7)
        # Map into a dense array indexed by node id (node_ids are 0..N-1 contiguous)
        self._scores = np.zeros(self.graph.n_nodes, dtype=np.float64)
        for n, s in pr.items():
            self._scores[n] = s
        logger.info("PageRank: top-5 nodes = %s",
                    np.argsort(self._scores)[::-1][:5].tolist())

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")
        self._ensure_scores()
        scores = self._scores
        best = max(valid_actions, key=lambda v: scores[v])
        self.increment_step()
        logger.debug("PageRank picked node=%d  score=%.6f", best, scores[best])
        return best


# --------------------------------------------------------------------------- #
# k-shell                                                                      #
# --------------------------------------------------------------------------- #


class KShellAgent(BaseAgent):
    """
    Top-k by k-core decomposition (core number).

    Kitsak et al. 2010 showed k-shell is a strong indicator of *spreading*
    capacity — nodes in the inner cores of the network tend to launch larger
    cascades, often outperforming degree centrality.
    """

    def __init__(
        self,
        config: AgentConfig,
        graph: GraphData,
    ) -> None:
        super().__init__(config)
        self.graph = graph
        self._scores: Optional[np.ndarray] = None

    def _ensure_scores(self) -> None:
        if self._scores is not None:
            return
        logger.info("k-shell: computing core numbers on undirected projection...")
        g = _build_nx_digraph(self.graph)
        # NetworkX's core_number requires a graph without self-loops + undirected
        u = g.to_undirected()
        u.remove_edges_from(nx.selfloop_edges(u))
        core = nx.core_number(u)
        # Tie-break by out-degree so that within the same shell we still prefer hubs
        self._scores = np.zeros(self.graph.n_nodes, dtype=np.float64)
        for n, c in core.items():
            self._scores[n] = float(c) + 1e-6 * float(self.graph.out_degree[n])
        max_core = int(self._scores.max())
        logger.info("k-shell: max core number = %d", max_core)

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")
        self._ensure_scores()
        scores = self._scores
        best = max(valid_actions, key=lambda v: scores[v])
        self.increment_step()
        logger.debug("k-shell picked node=%d  score=%.4f", best, scores[best])
        return best


# --------------------------------------------------------------------------- #
# Betweenness                                                                  #
# --------------------------------------------------------------------------- #


class BetweennessAgent(BaseAgent):
    """
    Top-k by betweenness centrality.

    Betweenness measures the fraction of shortest paths passing through a
    node — high-BC nodes are bridges between regions of the graph and tend
    to be key cascade spreaders.

    Note: O(N·M) exact; for graphs > ~10k nodes we use approximate sampling.
    """

    def __init__(
        self,
        config: AgentConfig,
        graph: GraphData,
        approx_threshold: int = 5000,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        self.graph = graph
        self.approx_threshold = int(approx_threshold)
        self.n_samples = int(n_samples)
        self._seed = seed
        self._scores: Optional[np.ndarray] = None

    def _ensure_scores(self) -> None:
        if self._scores is not None:
            return
        n = self.graph.n_nodes
        g = _build_nx_digraph(self.graph)

        if n > self.approx_threshold:
            logger.info("Betweenness: approximating with k=%d source samples (n=%d)...",
                        self.n_samples, n)
            bc = nx.betweenness_centrality(
                g, k=min(self.n_samples, n), seed=self._seed, normalized=True,
            )
        else:
            logger.info("Betweenness: computing exact (n=%d)...", n)
            bc = nx.betweenness_centrality(g, normalized=True)

        self._scores = np.zeros(n, dtype=np.float64)
        for node, s in bc.items():
            self._scores[node] = s
        logger.info("Betweenness: top-5 nodes = %s",
                    np.argsort(self._scores)[::-1][:5].tolist())

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")
        self._ensure_scores()
        scores = self._scores
        best = max(valid_actions, key=lambda v: scores[v])
        self.increment_step()
        logger.debug("Betweenness picked node=%d  score=%.6f", best, scores[best])
        return best


# --------------------------------------------------------------------------- #
# Degree Discount                                                              #
# --------------------------------------------------------------------------- #


class DegreeDiscountAgent(BaseAgent):
    """
    Degree Discount heuristic (Chen, Wang, Yang 2009).

    Recompute candidate scores at every step:
        score(v) = d_v − 2·t_v − (d_v − t_v)·t_v·p
    where
        d_v = total (out + in) degree of v,
        t_v = number of v's neighbours already in the seed set,
        p   = mean edge influence probability.

    Intuition: as more of v's neighbours are seeded, the *additional* spread
    we expect from seeding v shrinks, because v's neighbourhood is already
    likely to be activated.

    Unlike pure degree, DegreeDiscount is *state-aware* and approximates the
    submodular discounting effect for free (no MC simulations).
    """

    def __init__(
        self,
        config: AgentConfig,
        graph: GraphData,
        ic: Optional[ICDiffusion] = None,
        propagation_p: Optional[float] = None,
    ) -> None:
        super().__init__(config)
        self.graph = graph
        # Combined adjacency (treat directed graph as undirected for "neighbour" count)
        self._adj: Dict[int, set] = {n: set() for n in graph.node_ids}
        for u, v in graph.edges:
            self._adj[u].add(v)
            self._adj[v].add(u)
        # Effective propagation prob: average edge prob from IC if supplied,
        # otherwise default to 0.01 (the value used in the original paper)
        if propagation_p is not None:
            self._p = float(propagation_p)
        elif ic is not None and ic.edge_probs:
            self._p = float(np.mean(list(ic.edge_probs.values())))
        else:
            self._p = 0.01
        # Combined degree (in + out, dedup'd by undirected projection)
        self._deg = np.array([len(self._adj[n]) for n in graph.node_ids],
                             dtype=np.float64)
        logger.info("DegreeDiscount: using propagation_p=%.4f", self._p)

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")
        seed_set = state.seed_set
        p = self._p

        best_node = -1
        best_score = -np.inf
        for v in valid_actions:
            d_v = self._deg[v]
            t_v = sum(1 for u in self._adj[v] if u in seed_set)
            score = d_v - 2.0 * t_v - (d_v - t_v) * t_v * p
            if score > best_score:
                best_score = score
                best_node = v

        self.increment_step()
        logger.debug("DegreeDiscount picked node=%d  score=%.3f  p=%.4f",
                     best_node, best_score, self._p)
        return best_node
