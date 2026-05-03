from influence_max.agents.base import BaseAgent
from influence_max.agents.bandit import LinUCBAgent
from influence_max.agents.celf import CELFAgent, CELFppAgent
from influence_max.agents.centrality import (
    BetweennessAgent,
    DegreeDiscountAgent,
    KShellAgent,
    PageRankAgent,
)
from influence_max.agents.community import LouvainAgent
from influence_max.agents.dqn import DQNAgent
from influence_max.agents.monte_carlo import (
    DegreeHeuristicAgent,
    GreedyMCAgent,
    RandomAgent,
    build_agent,
)
from influence_max.agents.ris import IMMAgent
from influence_max.agents.s2v_dqn import S2VDQNAgent
from influence_max.agents.sarsa import SARSAAgent

__all__ = [
    "BaseAgent",
    # Baselines
    "GreedyMCAgent",
    "RandomAgent",
    "DegreeHeuristicAgent",
    # Lazy greedy
    "CELFAgent",
    "CELFppAgent",
    # RIS
    "IMMAgent",
    # Centrality / heuristics
    "PageRankAgent",
    "KShellAgent",
    "BetweennessAgent",
    "DegreeDiscountAgent",
    # Community
    "LouvainAgent",
    # Learning / DL
    "DQNAgent",
    "SARSAAgent",
    "LinUCBAgent",
    "S2VDQNAgent",
    # Factory
    "build_agent",
]
