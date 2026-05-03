"""
Independent Cascade (IC) diffusion model using learned GNN embeddings.

P(u -> v) = sigmoid(H_u · H_v)

The ICDiffusion class pre-caches edge probabilities at construction time
so that cascade simulations are fast numpy operations (no repeated tensor ops).

MonteCarloEstimator wraps ICDiffusion to run N parallel cascade simulations
and return:
  - mean activated count
  - std of activated count
  - union of all activated node sets (used by the env for action masking)
  - per-simulation activated sets (for downstream value estimation)
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import torch

from influence_max.config import ICConfig, resolve_device
from influence_max.graph import GraphData

logger = logging.getLogger(__name__)


class ICDiffusion:
    """
    Independent Cascade diffusion model backed by GNN dot-product probabilities.

    Parameters
    ----------
    graph       : GraphData – directed graph adjacency structures
    config      : ICConfig
    embeddings  : optional pre-loaded tensor; if None, loads from config.embeddings_path

    Attributes
    ----------
    edge_probs : Dict[(u, v) -> float]
        Pre-cached influence probability for each directed edge.
        Edges below config.prob_threshold are omitted (treated as 0).
    """

    def __init__(
        self,
        graph: GraphData,
        config: ICConfig,
        embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        self.graph = graph
        self.config = config

        self.device = resolve_device(config.device)
        logger.info("ICDiffusion using device: %s", self.device)

        if embeddings is None:
            embeddings = torch.load(config.embeddings_path, map_location="cpu", weights_only=True)
        # Move embedding matrix to the compute device for fast batched dot-products
        self._H: torch.Tensor = embeddings.float().to(self.device)

        logger.info(
            "Building edge probability cache for %d edges (threshold=%.1e)...",
            graph.n_edges,
            config.prob_threshold,
        )
        self.edge_probs: Dict[Tuple[int, int], float] = self._build_edge_prob_cache()
        logger.info("Edge prob cache built: %d edges retained.", len(self.edge_probs))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_edge_prob_cache(self) -> Dict[Tuple[int, int], float]:
        """
        Vectorised computation of P(u->v) for every edge in the graph.

        Processes edges in batches to keep memory usage bounded on large graphs.
        """
        edges = self.graph.edges
        batch_size = 4096
        probs: Dict[Tuple[int, int], float] = {}

        with torch.no_grad():
            for start in range(0, len(edges), batch_size):
                batch = edges[start : start + batch_size]
                us = torch.tensor([e[0] for e in batch], dtype=torch.long, device=self.device)
                vs = torch.tensor([e[1] for e in batch], dtype=torch.long, device=self.device)
                h_u = self._H[us]  # (B, D)
                h_v = self._H[vs]  # (B, D)
                # Move back to CPU for numpy; MPS/CUDA tensors can't call .numpy() directly
                p = torch.sigmoid((h_u * h_v).sum(dim=-1)).cpu().numpy()  # (B,)

                for (u, v), prob in zip(batch, p):
                    if prob >= self.config.prob_threshold:
                        probs[(u, v)] = float(prob)

        return probs

    def influence_probability(self, u: int, v: int) -> float:
        """Return the cached P(u->v), or 0 if below threshold / not an edge."""
        return self.edge_probs.get((u, v), 0.0)

    # ------------------------------------------------------------------ #
    # Single cascade simulation                                            #
    # ------------------------------------------------------------------ #

    def run_cascade(
        self,
        seed_set: FrozenSet[int],
        rng: np.random.Generator,
    ) -> FrozenSet[int]:
        """
        Run one IC cascade from *seed_set* and return all activated nodes
        (including the seeds themselves).

        Algorithm
        ---------
        BFS layer-by-layer.  For each newly activated node u, try each
        out-neighbour v not yet activated: activate v with prob P(u->v).
        """
        activated: Set[int] = set(seed_set)
        frontier: List[int] = list(seed_set)

        while frontier:
            next_frontier: List[int] = []
            for u in frontier:
                for v in self.graph.successors(u):
                    if v in activated:
                        continue
                    p = self.edge_probs.get((u, v), 0.0)
                    if p > 0.0 and rng.random() < p:
                        activated.add(v)
                        next_frontier.append(v)
            frontier = next_frontier

        return frozenset(activated)


class MonteCarloEstimator:
    """
    Estimates influence spread of a seed set via repeated IC simulations.

    Parameters
    ----------
    ic      : ICDiffusion model
    config  : ICConfig (n_simulations, seed)
    """

    def __init__(self, ic: ICDiffusion, config: ICConfig) -> None:
        self.ic = ic
        self.config = config
        self._base_rng = np.random.default_rng(config.seed)

    def estimate(
        self,
        seed_set: FrozenSet[int],
        n_simulations: Optional[int] = None,
    ) -> "MCResult":
        """
        Run N IC cascades from *seed_set*.

        Parameters
        ----------
        seed_set      : nodes already selected as seeds
        n_simulations : override for config.n_simulations

        Returns
        -------
        MCResult with:
          - mean_spread  : average |activated| across simulations
          - std_spread   : std of |activated|
          - union_activated : union of all activated sets (used for action masking)
          - sim_results  : list of frozensets from each simulation
        """
        n = n_simulations if n_simulations is not None else self.config.n_simulations

        if not seed_set:
            empty = frozenset()
            return MCResult(
                seed_set=seed_set,
                mean_spread=0.0,
                std_spread=0.0,
                union_activated=empty,
                sim_results=[empty] * n,
            )

        # Each simulation gets its own RNG stream derived from the base seed
        # to ensure reproducibility while allowing parallelism in the future.
        sim_results: List[FrozenSet[int]] = []
        counts: List[int] = []
        union: Set[int] = set()

        for i in range(n):
            rng = np.random.default_rng([self.config.seed, i] if self.config.seed is not None else None)
            activated = self.ic.run_cascade(seed_set, rng)
            sim_results.append(activated)
            counts.append(len(activated))
            union.update(activated)

        counts_arr = np.array(counts, dtype=np.float64)
        return MCResult(
            seed_set=seed_set,
            mean_spread=float(counts_arr.mean()),
            std_spread=float(counts_arr.std()),
            union_activated=frozenset(union),
            sim_results=sim_results,
        )


class MCResult:
    """
    Container for Monte Carlo influence estimation results.

    Attributes
    ----------
    seed_set        : the seed set that was evaluated
    mean_spread     : mean number of activated nodes across simulations
    std_spread      : std dev
    union_activated : union of activated nodes across ALL simulations
                      – used by the env to build the action mask
                      (conservative: excludes nodes activated in *any* sim)
    sim_results     : per-simulation activated frozensets
    """

    __slots__ = ("seed_set", "mean_spread", "std_spread", "union_activated", "sim_results")

    def __init__(
        self,
        seed_set: FrozenSet[int],
        mean_spread: float,
        std_spread: float,
        union_activated: FrozenSet[int],
        sim_results: List[FrozenSet[int]],
    ) -> None:
        self.seed_set = seed_set
        self.mean_spread = mean_spread
        self.std_spread = std_spread
        self.union_activated = union_activated
        self.sim_results = sim_results

    def __repr__(self) -> str:
        return (
            f"MCResult(seeds={len(self.seed_set)}, "
            f"spread={self.mean_spread:.2f}±{self.std_spread:.2f}, "
            f"union_activated={len(self.union_activated)})"
        )
