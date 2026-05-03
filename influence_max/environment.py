"""
RL environment for influence maximization.

State: S_t = (seed_set, budget_remaining)
  - seed_set       : frozenset of nodes already chosen as seeds
  - budget         : remaining seed selections

Activated nodes are NOT part of the state.  They are computed on-demand by
running the IC model on the current seed_set.  The environment exposes them
via the `info` dict returned from `step()` and via `action_mask()` so that the
policy can filter out nodes it must not pick (seeds + activated).

Reward: r_t = mean_spread(seed_t+1) - mean_spread(seed_t)
           = marginal influence gain of adding action a_t.

The interface mirrors OpenAI Gym (reset / step / action_mask) so that any
future agent (Q-learning, DQN, etc.) can drop in without env changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from influence_max.config import EnvConfig, ICConfig
from influence_max.graph import GraphData
from influence_max.ic_model import ICDiffusion, MCResult, MonteCarloEstimator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class State:
    """
    Immutable RL state.

    seed_set : frozenset[int]  – nodes chosen as seeds so far
    budget   : int             – remaining seed selections
    """

    seed_set: FrozenSet[int]
    budget: int

    def __repr__(self) -> str:
        return f"State(seeds={sorted(self.seed_set)}, budget={self.budget})"


# --------------------------------------------------------------------------- #
# Transition named tuple                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class Transition:
    """
    A single environment transition, used by agents for learning updates.

    Compatible with replay buffers for Q-learning / DQN.
    """

    state: State
    action: int
    reward: float
    next_state: State
    done: bool
    info: Dict[str, Any]


# --------------------------------------------------------------------------- #
# Environment                                                                  #
# --------------------------------------------------------------------------- #


class InfluenceMaxEnv:
    """
    Gym-style environment for the influence maximization problem.

    Parameters
    ----------
    graph   : GraphData
    ic      : ICDiffusion   – pre-built, shared across episodes
    mc      : MonteCarloEstimator
    config  : EnvConfig

    Usage
    -----
    env = InfluenceMaxEnv(graph, ic, mc, config)
    state = env.reset()
    while not done:
        valid = env.action_mask(state, info["union_activated"])
        action = agent.select_action(state, valid)
        state, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        graph: GraphData,
        ic: ICDiffusion,
        mc: MonteCarloEstimator,
        config: EnvConfig,
    ) -> None:
        self.graph = graph
        self.ic = ic
        self.mc = mc
        self.config = config

        self._state: Optional[State] = None
        # Cache the MC result for the *current* seed set to avoid recomputing
        # it twice (once for the old state reward, once for the new state mask).
        self._current_mc: Optional[MCResult] = None
        self._episode_transitions: List[Transition] = []

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[State, Dict[str, Any]]:
        """
        Start a new episode.

        Returns
        -------
        state : initial State (empty seed set, full budget)
        info  : dict with "union_activated" = frozenset() (nothing activated yet)
        """
        self._state = State(seed_set=frozenset(), budget=self.config.budget)
        self._current_mc = self.mc.estimate(frozenset())
        self._episode_transitions = []

        info: Dict[str, Any] = {
            "union_activated": self._current_mc.union_activated,
            "mean_spread": self._current_mc.mean_spread,
            "mc_result": self._current_mc,
        }
        logger.debug("Episode reset. %s", self._state)
        return self._state, info

    # ------------------------------------------------------------------ #
    # Step                                                                 #
    # ------------------------------------------------------------------ #

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Add *action* to the seed set and transition to the next state.

        Parameters
        ----------
        action : node id to add to the seed set

        Returns
        -------
        next_state : State
        reward     : marginal influence gain (float)
        done       : True when budget is exhausted
        info       : dict containing MC results and activated set for masking
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        state = self._state
        prev_mc = self._current_mc

        # Validate action
        if action in state.seed_set:
            raise ValueError(f"Node {action} is already in the seed set.")
        if state.budget <= 0:
            raise RuntimeError("Budget exhausted; episode is done.")

        # --- Transition ------------------------------------------------ #
        new_seed_set = state.seed_set | frozenset([action])
        new_budget = state.budget - 1
        next_state = State(seed_set=new_seed_set, budget=new_budget)

        # --- Reward: marginal MC influence gain ------------------------- #
        next_mc = self.mc.estimate(new_seed_set)
        reward = next_mc.mean_spread - prev_mc.mean_spread

        done = new_budget == 0

        info: Dict[str, Any] = {
            "union_activated": next_mc.union_activated,
            "mean_spread": next_mc.mean_spread,
            "std_spread": next_mc.std_spread,
            "mc_result": next_mc,
            "prev_mean_spread": prev_mc.mean_spread,
        }

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
        )
        self._episode_transitions.append(transition)

        self._state = next_state
        self._current_mc = next_mc

        logger.debug(
            "step: action=%d  reward=%.3f  spread=%.2f  done=%s",
            action,
            reward,
            next_mc.mean_spread,
            done,
        )
        return next_state, reward, done, info

    # ------------------------------------------------------------------ #
    # Action mask                                                          #
    # ------------------------------------------------------------------ #

    def action_mask(
        self,
        state: State,
        union_activated: FrozenSet[int],
    ) -> np.ndarray:
        """
        Return a boolean array of shape (n_nodes,) where True means the node
        is a valid action (can be selected as a seed).

        Excluded nodes:
          1. Already in seed_set
          2. Already activated by the current seed_set (if config.mask_activated)

        Parameters
        ----------
        state           : current State
        union_activated : union of activated nodes from MC on current seed_set
        """
        mask = np.ones(self.graph.n_nodes, dtype=bool)
        node_ids = self.graph.node_ids  # sorted list, index == position

        # Build a fast index map: node_id -> position
        # (p2p-Gnutella08 nodes are 0-based so position == node_id, but we
        #  handle the general case for portability.)
        excluded = set(state.seed_set)
        if self.config.mask_activated:
            excluded |= set(union_activated)

        for node in excluded:
            # node_ids are sorted; for 0-based graphs node == index
            if 0 <= node < self.graph.n_nodes:
                mask[node] = False

        return mask

    def valid_actions(
        self,
        state: State,
        union_activated: FrozenSet[int],
    ) -> List[int]:
        """Return sorted list of valid action node ids."""
        mask = self.action_mask(state, union_activated)
        return [self.graph.node_ids[i] for i, m in enumerate(mask) if m]

    # ------------------------------------------------------------------ #
    # Episode history                                                       #
    # ------------------------------------------------------------------ #

    @property
    def episode_transitions(self) -> List[Transition]:
        return list(self._episode_transitions)

    @property
    def current_state(self) -> Optional[State]:
        return self._state

    @property
    def current_mc(self) -> Optional[MCResult]:
        return self._current_mc
