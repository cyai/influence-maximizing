"""
Centralized configuration dataclasses for the influence maximization RL pipeline.
All hyperparameters live here so that swapping agents (MC -> Q-learning -> DQN)
requires only a new AgentConfig section, not changes to env or IC model code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve the compute device string to a torch.device.

    Priority when device="auto":  MPS  >  CUDA  >  CPU

    MPS  – Apple Silicon GPU (M1/M2/M3/M4 via Metal Performance Shaders)
    CUDA – NVIDIA GPU
    CPU  – fallback
    """
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class GraphConfig:
    data_path: str = "data/p2p-Gnutella08.txt"
    # Graph is directed; edges are stored as-is for IC diffusion
    directed: bool = True


@dataclass
class ICConfig:
    """Independent Cascade diffusion model configuration."""

    embeddings_path: str = "node_embeddings.pt"
    # Number of Monte Carlo rollouts per influence estimate
    n_simulations: int = 100
    # Threshold below which edge probabilities are treated as 0 (speed optimisation)
    prob_threshold: float = 1e-4
    # Random seed for reproducible MC rollouts (None = non-deterministic)
    seed: Optional[int] = 42
    # Compute device for embedding dot-products: "auto" | "mps" | "cuda" | "cpu"
    # "auto" picks MPS > CUDA > CPU automatically.
    device: str = "auto"


@dataclass
class EnvConfig:
    """RL environment configuration."""

    budget: int = 10
    # Whether the action space excludes already-activated nodes (always True in our design)
    mask_activated: bool = True


@dataclass
class AgentConfig:
    """
    Agent / policy configuration.

    strategy options:
      - "greedy_mc"  : argmax marginal MC influence gain (GreedyMCAgent)
      - "random"     : uniform random over valid actions
      - "degree"     : pick highest out-degree node not yet seeded/activated
      - "dqn"        : Deep Q-Network (function approximation, replay buffer)
      - "sarsa"      : On-policy TD learning with linear function approximation
      - "linucb"     : Linear contextual bandit (UCB exploration)
    """

    strategy: Literal[
        "greedy_mc",
        "random",
        "degree",
        "dqn",
        "sarsa",
        "linucb",
        "celf",
        "celfpp",
        "imm",
        "pagerank",
        "kshell",
        "betweenness",
        "degree_discount",
        "louvain",
        "s2v_dqn",
    ] = "greedy_mc"
    # MC sims used *inside* the agent when evaluating candidate actions
    # (may differ from ICConfig.n_simulations used for reward computation)
    mc_rollouts: int = 50

    # --- Q-learning / SARSA shared hyperparameters ---
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500

    # --- DQN-specific ---
    replay_buffer_size: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 10

    # --- LinUCB-specific ---
    linucb_alpha: float = 1.0  # UCB exploration coefficient
    linucb_lambda: float = 1.0  # Ridge regularisation strength

    # --- IMM-specific ---
    imm_epsilon: float = 0.1  # approximation error parameter
    imm_delta: float = 0.0  # 0 → defaults to 1/N inside agent
    imm_theta_max: int = 200_000  # cap on RR sets for runtime safety

    # --- Centrality-specific ---
    pagerank_alpha: float = 0.85
    betweenness_n_samples: int = 1000  # used for graphs > 5000 nodes
    dd_propagation_p: float = 0.0  # 0 → use mean(IC edge prob) inside agent

    # --- Community (Louvain) ---
    community_resolution: float = 1.0

    # --- S2V-DQN-specific ---
    s2v_t_iters: int = 4
    s2v_hidden_dim: int = 64

    # --- Embedding-based agents need access to GNN embeddings ---
    embeddings_path: str = "node_embeddings.pt"


@dataclass
class TrainConfig:
    """Top-level training run configuration."""

    graph: GraphConfig = field(default_factory=GraphConfig)
    ic: ICConfig = field(default_factory=ICConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    n_episodes: int = 1
    results_dir: str = "results"
    log_level: str = "INFO"
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(
            graph=GraphConfig(**d.get("graph", {})),
            ic=ICConfig(**d.get("ic", {})),
            env=EnvConfig(**d.get("env", {})),
            agent=AgentConfig(**d.get("agent", {})),
            n_episodes=d.get("n_episodes", 1),
            results_dir=d.get("results_dir", "results"),
            log_level=d.get("log_level", "INFO"),
            seed=d.get("seed", 42),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        d = json.loads(Path(path).read_text())
        return cls.from_dict(d)
