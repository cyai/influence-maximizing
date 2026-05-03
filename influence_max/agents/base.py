"""
Abstract base classes for RL agents in the influence maximization pipeline.

All concrete agents (Monte Carlo, Q-learning, DQN, etc.) must implement
BaseAgent.  The interface is intentionally minimal so that new algorithms
can be dropped in without changing the environment or training loop.

Design contract
---------------
- select_action : choose an action given state + valid action list
- update        : incorporate one Transition into the agent's learning signal
                  (no-op for non-learning agents like GreedyMCAgent)
- save / load   : persist and restore agent parameters
- reset_episode : called at the start of each episode (e.g. clear eligibility
                  traces, episode buffers); no-op by default
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from influence_max.config import AgentConfig
from influence_max.environment import State, Transition

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all influence-maximization RL agents.

    Parameters
    ----------
    config : AgentConfig
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._step: int = 0  # global step counter, useful for epsilon schedules etc.

    # ------------------------------------------------------------------ #
    # Core interface (must be implemented)                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        **kwargs: Any,
    ) -> int:
        """
        Choose the next seed node.

        Parameters
        ----------
        state         : current RL state (seed_set, budget)
        valid_actions : list of node ids that are legal to select
                        (already filtered: no seeds, no activated nodes)
        **kwargs      : agent-specific extras (e.g. mc_result for MC agents)

        Returns
        -------
        int : chosen node id
        """

    # ------------------------------------------------------------------ #
    # Learning interface (no-op for non-learning agents)                  #
    # ------------------------------------------------------------------ #

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        """
        Incorporate a single environment Transition.

        Returns an optional dict of training metrics (e.g. {"loss": 0.12}).
        Non-learning agents return None.
        """
        return None

    def update_batch(self, transitions: List[Transition]) -> Optional[Dict[str, float]]:
        """
        Batch update (used by DQN / replay-buffer agents).
        Default: call update() for each transition sequentially.
        """
        metrics_list = [self.update(t) for t in transitions]
        valid = [m for m in metrics_list if m is not None]
        if not valid:
            return None
        # Average scalar metrics across the batch
        keys = valid[0].keys()
        return {k: float(np.mean([m[k] for m in valid])) for k in keys}

    # ------------------------------------------------------------------ #
    # Episode lifecycle hooks                                              #
    # ------------------------------------------------------------------ #

    def reset_episode(self) -> None:
        """Called at the start of each new episode. Override as needed."""

    def end_episode(self, total_reward: float, episode_transitions: List[Transition]) -> None:
        """
        Called at episode end.  Useful for Monte Carlo return computation,
        logging, etc.  Override as needed.
        """

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """
        Persist agent state to *path*.

        Base implementation saves AgentConfig as JSON.
        Override to also save model weights (Q-table, neural net, etc.).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "agent_class": self.__class__.__name__,
            "config": self.config.__dict__,
            "step": self._step,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Agent saved to %s", path)

    def load(self, path: str | Path) -> None:
        """
        Restore agent state from *path*.

        Base implementation restores the step counter.
        Override to also load model weights.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        self._step = data.get("step", 0)
        logger.info("Agent loaded from %s (step=%d)", path, self._step)

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def increment_step(self) -> None:
        self._step += 1

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(strategy={self.config.strategy}, step={self._step})"
