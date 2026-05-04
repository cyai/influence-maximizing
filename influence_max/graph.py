"""
Graph loading and adjacency utilities for the influence maximization pipeline.

Supports two on-disk formats:
  - Whitespace-separated edge list (SNAP .txt)   ← p2p-Gnutella08
  - Comma-separated signed ratings (SNAP .csv)   ← soc-sign-bitcoin-alpha
    Columns: SOURCE,TARGET,RATING,TIME
    Only edges with RATING > 0 are retained for influence propagation.

Node ids are re-indexed to 0..N-1 when the original ids are not contiguous.
soc-sign-bitcoin-alpha —
3,783 nodes, 24,186 edges"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

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
    edges   : list of (u, v) tuples  (0-indexed)
    out_neighbors : dict[node -> list[node]]  (successors in directed graph)
    in_neighbors  : dict[node -> list[node]]  (predecessors)
    out_degree    : np.ndarray shape (n_nodes,)
    in_degree     : np.ndarray shape (n_nodes,)
    node_ids      : sorted list of all node ids (0..N-1 after re-indexing)
    id_map        : original_id -> contiguous_index  (identity when not needed)
    """

    n_nodes: int
    n_edges: int
    edges: List[Tuple[int, int]]
    out_neighbors: Dict[int, List[int]]
    in_neighbors: Dict[int, List[int]]
    out_degree: np.ndarray
    in_degree: np.ndarray
    node_ids: List[int]
    id_map: Optional[Dict[int, int]] = None   # None means identity mapping

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
    Auto-detect format from file extension and load the graph.

    - ``.txt`` / ``.gz`` (whitespace)  → SNAP edge-list (p2p-Gnutella08 style)
    - ``.csv``                          → signed ratings CSV (Bitcoin Alpha style)
      Columns: SOURCE,TARGET,RATING,TIME   (may have header row)
      Only RATING > 0 edges are kept for influence propagation.

    Node ids are always re-indexed to 0..N-1.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_signed_csv(path)
    else:
        return _load_txt_edgelist(path)


# --------------------------------------------------------------------------- #
# SNAP whitespace edge-list loader  (p2p-Gnutella08 and similar)              #
# --------------------------------------------------------------------------- #

def _load_txt_edgelist(path: Path) -> GraphData:
    raw_edges: List[Tuple[int, int]] = []
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
                continue
            raw_edges.append((u, v))
            node_set.update((u, v))

    return _build_graph_data(raw_edges, node_set, path)


# --------------------------------------------------------------------------- #
# Signed CSV loader  (soc-sign-bitcoin-alpha and similar)                     #
# --------------------------------------------------------------------------- #

def _load_signed_csv(path: Path, min_rating: int = 1) -> GraphData:
    """
    Parse SOURCE,TARGET,RATING,TIME CSV (Bitcoin Alpha / OTC format).

    Only edges with RATING >= min_rating are kept so that influence
    propagates only through *trust* relationships.

    The original node ids (arbitrary integers starting at 1) are
    re-indexed to 0..N-1.
    """
    raw_edges: List[Tuple[int, int]] = []
    node_set: Set[int] = set()
    skipped_negative = 0

    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip header rows (first token not a digit)
            if not row[0].strip().lstrip("-").isdigit():
                continue
            if len(row) < 3:
                continue
            src, tgt, rating = int(row[0]), int(row[1]), int(row[2])
            if src == tgt:
                continue
            if rating < min_rating:
                skipped_negative += 1
                continue
            raw_edges.append((src, tgt))
            node_set.update((src, tgt))

    logger.info(
        "Bitcoin CSV: %d positive-rating edges kept, %d negative/neutral skipped",
        len(raw_edges), skipped_negative,
    )
    return _build_graph_data(raw_edges, node_set, path)


# --------------------------------------------------------------------------- #
# Shared adjacency builder                                                     #
# --------------------------------------------------------------------------- #

def _build_graph_data(
    raw_edges: List[Tuple[int, int]],
    node_set: Set[int],
    path: Path,
) -> GraphData:
    """Re-index node ids to 0..N-1 and build all adjacency structures."""
    # Contiguous re-indexing (a no-op for 0-based graphs like p2p-Gnutella08)
    sorted_nodes = sorted(node_set)
    id_map: Dict[int, int] = {orig: idx for idx, orig in enumerate(sorted_nodes)}
    identity = all(orig == idx for orig, idx in id_map.items())

    edges = [(id_map[u], id_map[v]) for u, v in raw_edges]
    n_nodes = len(sorted_nodes)
    n_edges = len(edges)
    node_ids = list(range(n_nodes))

    out_neighbors: Dict[int, List[int]] = {n: [] for n in node_ids}
    in_neighbors:  Dict[int, List[int]] = {n: [] for n in node_ids}
    for u, v in edges:
        out_neighbors[u].append(v)
        in_neighbors[v].append(u)

    out_degree = np.array([len(out_neighbors[n]) for n in node_ids], dtype=np.int32)
    in_degree  = np.array([len(in_neighbors[n])  for n in node_ids], dtype=np.int32)

    logger.info(
        "Loaded graph: %d nodes, %d directed edges from %s",
        n_nodes, n_edges, path,
    )

    return GraphData(
        n_nodes=n_nodes,
        n_edges=n_edges,
        edges=edges,
        out_neighbors=out_neighbors,
        in_neighbors=in_neighbors,
        out_degree=out_degree,
        in_degree=in_degree,
        node_ids=node_ids,
        id_map=None if identity else id_map,
    )
