"""
Graph loading and adjacency utilities for the influence maximization pipeline.

Loads the directed p2p-Gnutella08 edge list and exposes:
  - NetworkX DiGraph for general use
  - Neighbour lookup dicts for fast IC cascade iteration
  - Degree arrays for degree-heuristic agents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """
    Lightweight directed graph representation optimised for IC diffusion.

    Attributes
    ----------
    n_nodes : int
    n_edges : int
    edges   : list of (u, v) tuples
    out_neighbors : dict[node -> list[node]]  (successors in directed graph)
    in_neighbors  : dict[node -> list[node]]  (predecessors)
    out_degree    : np.ndarray shape (n_nodes,)
    in_degree     : np.ndarray shape (n_nodes,)
    node_ids      : sorted list of all node ids
    """

    n_nodes: int
    n_edges: int
    edges: List[Tuple[int, int]]
    out_neighbors: Dict[int, List[int]]
    in_neighbors: Dict[int, List[int]]
    out_degree: np.ndarray
    in_degree: np.ndarray
    node_ids: List[int]

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    def successors(self, node: int) -> List[int]:
        return self.out_neighbors.get(node, [])

    def predecessors(self, node: int) -> List[int]:
        return self.in_neighbors.get(node, [])

    def top_k_by_out_degree(self, k: int) -> List[int]:
        """Return the k nodes with highest out-degree."""
        indices = np.argpartition(self.out_degree, -k)[-k:]
        return sorted(indices.tolist(), key=lambda n: self.out_degree[n], reverse=True)

    def nodes_as_set(self) -> FrozenSet[int]:
        return frozenset(self.node_ids)


def load_graph(data_path: str | Path) -> GraphData:
    """
    Parse a whitespace-separated edge-list file (SNAP format).

    Lines starting with '#' are treated as comments and skipped.
    Nodes are expected to be non-negative integers.

    Returns a :class:`GraphData` with pre-built adjacency structures.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    edges: List[Tuple[int, int]] = []
    node_set: Set[int] = set()

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue  # skip self-loops
            edges.append((u, v))
            node_set.update((u, v))

    node_ids = sorted(node_set)
    n_nodes = len(node_ids)
    n_edges = len(edges)

    logger.info("Loaded graph: %d nodes, %d directed edges from %s", n_nodes, n_edges, path)

    # Build contiguous index mapping if node ids are not already 0..N-1
    # p2p-Gnutella08 uses 0-based ids that match embedding indices directly.
    out_neighbors: Dict[int, List[int]] = {n: [] for n in node_ids}
    in_neighbors: Dict[int, List[int]] = {n: [] for n in node_ids}

    for u, v in edges:
        out_neighbors[u].append(v)
        in_neighbors[v].append(u)

    out_degree = np.array([len(out_neighbors[n]) for n in node_ids], dtype=np.int32)
    in_degree = np.array([len(in_neighbors[n]) for n in node_ids], dtype=np.int32)

    return GraphData(
        n_nodes=n_nodes,
        n_edges=n_edges,
        edges=edges,
        out_neighbors=out_neighbors,
        in_neighbors=in_neighbors,
        out_degree=out_degree,
        in_degree=in_degree,
        node_ids=node_ids,
    )
