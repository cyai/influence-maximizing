"""
influence_max – RL pipeline for influence maximization.

Modules
-------
config      : dataclasses for all hyperparameters
graph       : graph loading and adjacency structures
ic_model    : IC diffusion model + Monte Carlo estimator
environment : gym-style RL environment (State, InfluenceMaxEnv)
agents/     : BaseAgent + concrete implementations
"""

__version__ = "0.1.0"
