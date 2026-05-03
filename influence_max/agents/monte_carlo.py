"""
Monte Carlo-based influence maximization agents and baselines.

Agents
------
GreedyMCAgent
    For each valid action, estimates the marginal influence gain by running
    MC simulations and picks the action with the highest gain.
    This is the "greedy + MC" strategy and serves as the main RL policy.

RandomAgent
    Selects a uniformly random valid action.
    Baseline: sanity check that GreedyMC is better than chance.

DegreeHeuristicAgent
    Selects the valid node with the highest out-degree.
    Baseline: classic influence-maximization heuristic, fast, no MC needed.

All agents implement BaseAgent and are selected via AgentConfig.strategy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional

import numpy as np

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig
from influence_max.environment import State, Transition
from influence_max.graph import GraphData
from influence_max.ic_model import ICDiffusion, MCResult, MonteCarloEstimator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Greedy Monte Carlo Agent                                                     #
# --------------------------------------------------------------------------- #


class GreedyMCAgent(BaseAgent):
    """
    Greedy influence maximization via Monte Carlo rollouts.

    For each valid candidate node, evaluates the marginal spread:
        delta(v) = MC_spread(seed_set ∪ {v}) - MC_spread(seed_set)

    Selects argmax_v delta(v).

    Parameters
    ----------
    config  : AgentConfig  (mc_rollouts controls rollouts for candidate eval)
    mc      : MonteCarloEstimator shared with the environment
    graph   : GraphData (not used for selection, kept for interface parity)
    """

    def __init__(
        self,
        config: AgentConfig,
        mc: MonteCarloEstimator,
        graph: GraphData,
    ) -> None:
        super().__init__(config)
        self.mc = mc
        self.graph = graph

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        current_mc_result: Optional[MCResult] = None,
        **kwargs: Any,
    ) -> int:
        """
        Choose the node that maximises expected marginal influence spread.

        Parameters
        ----------
        state             : current State (seed_set, budget)
        valid_actions     : list of candidate node ids (pre-filtered)
        current_mc_result : MCResult for the *current* seed_set, passed in from
                            the env so we don't recompute it.  If None, we run
                            a fresh estimate.

        Returns
        -------
        int : node with highest estimated marginal gain
        """
        if not valid_actions:
            raise ValueError("No valid actions available.")

        # Current spread (baseline)
        if current_mc_result is not None:
            base_spread = current_mc_result.mean_spread
        else:
            base_result = self.mc.estimate(
                state.seed_set, n_simulations=self.config.mc_rollouts
            )
            base_spread = base_result.mean_spread

        best_action: int = valid_actions[0]
        best_gain: float = -float("inf")

        logger.debug("Evaluating %d candidate actions...", len(valid_actions))

        for candidate in valid_actions:
            new_seeds = state.seed_set | frozenset([candidate])
            result = self.mc.estimate(new_seeds, n_simulations=self.config.mc_rollouts)
            gain = result.mean_spread - base_spread
            logger.debug("  node=%d  marginal_gain=%.3f", candidate, gain)
            if gain > best_gain:
                best_gain = gain
                best_action = candidate

        self.increment_step()
        logger.info(
            "GreedyMC selected node=%d  marginal_gain=%.3f  (from %d candidates)",
            best_action,
            best_gain,
            len(valid_actions),
        )
        return best_action

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        # Greedy MC is a planning agent — no model to update.
        return None


# --------------------------------------------------------------------------- #
# Random Baseline                                                              #
# --------------------------------------------------------------------------- #


class RandomAgent(BaseAgent):
    """
    Selects a uniformly random valid action.

    Useful as a lower-bound baseline.
    """

    def __init__(self, config: AgentConfig, seed: Optional[int] = None) -> None:
        super().__init__(config)
        self._rng = np.random.default_rng(seed)

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")
        action = int(self._rng.choice(valid_actions))
        self.increment_step()
        logger.info("Random selected node=%d", action)
        return action


# --------------------------------------------------------------------------- #
# Degree Heuristic Baseline                                                    #
# --------------------------------------------------------------------------- #


class DegreeHeuristicAgent(BaseAgent):
    """
    Selects the valid node with the highest out-degree.

    Classic IM heuristic.  Fast (O(k) per step) and surprisingly competitive
    on scale-free networks.
    """

    def __init__(self, config: AgentConfig, graph: GraphData) -> None:
        super().__init__(config)
        self.graph = graph

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")

        best_node = max(valid_actions, key=lambda v: self.graph.out_degree[v])
        self.increment_step()
        logger.info(
            "DegreeHeuristic selected node=%d  out_degree=%d",
            best_node,
            self.graph.out_degree[best_node],
        )
        return best_node


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #


def build_agent(
    config: AgentConfig,
    mc: MonteCarloEstimator,
    graph: GraphData,
    budget: int,
    device: str = "auto",
    seed: Optional[int] = None,
    ic: Optional[ICDiffusion] = None,
) -> BaseAgent:
    """
    Construct the agent specified by config.strategy.

    Parameters
    ----------
    config : AgentConfig
    mc     : MonteCarloEstimator (shared with env)
    graph  : GraphData
    budget : per-episode seed budget (needed by embedding-based agents
             to normalise the budget feature; also by IMM, Louvain)
    device : compute device for embedding-based agents
    seed   : optional RNG seed for stochastic agents
    ic     : ICDiffusion model — required for IMM, S2V-DQN, and used as a
             default propagation-prob source by DegreeDiscount

    Returns
    -------
    BaseAgent subclass instance
    """
    strategy = config.strategy
    if strategy == "greedy_mc":
        return GreedyMCAgent(config, mc, graph)
    elif strategy == "random":
        return RandomAgent(config, seed=seed)
    elif strategy == "degree":
        return DegreeHeuristicAgent(config, graph)

    # --- Lazy greedy: CELF / CELF++ ---
    elif strategy in {"celf", "celfpp"}:
        from influence_max.agents.celf import CELFAgent, CELFppAgent

        if strategy == "celf":
            return CELFAgent(config, mc, graph)
        return CELFppAgent(config, mc, graph)

    # --- Reverse Influence Sampling: IMM ---
    elif strategy == "imm":
        from influence_max.agents.ris import IMMAgent

        if ic is None:
            raise ValueError(
                "IMM requires the ICDiffusion model — pass `ic=` to build_agent."
            )
        delta = config.imm_delta if config.imm_delta > 0 else None
        return IMMAgent(
            config,
            graph=graph,
            ic=ic,
            budget=budget,
            epsilon=config.imm_epsilon,
            delta=delta,
            theta_max=config.imm_theta_max,
            seed=seed,
        )

    # --- Centrality / heuristics ---
    elif strategy in {"pagerank", "kshell", "betweenness", "degree_discount"}:
        from influence_max.agents.centrality import (
            BetweennessAgent,
            DegreeDiscountAgent,
            KShellAgent,
            PageRankAgent,
        )

        if strategy == "pagerank":
            return PageRankAgent(config, graph, alpha=config.pagerank_alpha)
        elif strategy == "kshell":
            return KShellAgent(config, graph)
        elif strategy == "betweenness":
            return BetweennessAgent(
                config,
                graph,
                n_samples=config.betweenness_n_samples,
                seed=seed,
            )
        else:  # degree_discount
            p = config.dd_propagation_p if config.dd_propagation_p > 0 else None
            return DegreeDiscountAgent(config, graph, ic=ic, propagation_p=p)

    # --- Community-based: Louvain ---
    elif strategy == "louvain":
        from influence_max.agents.community import LouvainAgent

        return LouvainAgent(
            config,
            mc,
            graph,
            budget=budget,
            resolution=config.community_resolution,
            seed=seed,
        )

    # --- Learning / DL with shared embeddings ---
    elif strategy in {"dqn", "sarsa", "linucb"}:
        # Lazy imports to avoid a hard torch dependency for the simple agents
        import torch
        from influence_max.agents.bandit import LinUCBAgent
        from influence_max.agents.dqn import DQNAgent
        from influence_max.agents.sarsa import SARSAAgent

        embeddings = torch.load(
            config.embeddings_path, map_location="cpu", weights_only=True
        ).float()

        if strategy == "dqn":
            return DQNAgent(
                config, embeddings, budget_max=budget, device=device, seed=seed
            )
        elif strategy == "sarsa":
            return SARSAAgent(
                config, embeddings, budget_max=budget, device=device, seed=seed
            )
        else:  # linucb
            return LinUCBAgent(
                config,
                embeddings,
                budget_max=budget,
                alpha=config.linucb_alpha,
                ridge_lambda=config.linucb_lambda,
                seed=seed,
            )

    # --- S2V-DQN: trains its own embeddings end-to-end ---
    elif strategy == "s2v_dqn":
        from influence_max.agents.s2v_dqn import S2VDQNAgent

        if ic is None:
            raise ValueError(
                "S2V-DQN requires the ICDiffusion model (for edge weights)."
            )
        return S2VDQNAgent(
            config,
            graph=graph,
            ic=ic,
            budget_max=budget,
            embed_dim=config.s2v_hidden_dim,
            t_iters=config.s2v_t_iters,
            device=device,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of: "
            "'greedy_mc', 'random', 'degree', "
            "'celf', 'celfpp', 'imm', "
            "'pagerank', 'kshell', 'betweenness', 'degree_discount', "
            "'louvain', 'dqn', 'sarsa', 'linucb', 's2v_dqn'."
        )
