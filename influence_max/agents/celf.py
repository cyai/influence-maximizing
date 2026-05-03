"""
Lazy-greedy influence maximization agents — CELF and CELF++.

Both produce the *same* seed set as the naive greedy baseline (`GreedyMCAgent`)
but evaluate far fewer (state, candidate) pairs by exploiting submodularity:
once a candidate's marginal gain is computed for some seed set S, its gain
for any superset S' ⊇ S can only be ≤ that value.

CELF — Leskovec et al. 2007
---------------------------
Maintain a max-heap of (mg, node, last_eval_iter).  For each round:

    while True:
        pop (mg, v, last_eval) from the heap
        if last_eval == |seeds|:        # already valid for current S
            return v                    # (submodularity → top is argmax)
        else:
            recompute mg = MC(seeds∪{v}) − MC(seeds)
            push (mg, v, |seeds|) back into heap

Worst case: O(N) MC calls per round. Best case: 1 MC call per round.
On real graphs CELF typically runs 100×–1000× faster than naive greedy.

CELF++ — Goyal et al. 2011
--------------------------
For each candidate v we additionally cache
    mg2(v) = MC(S ∪ {prev_best, v}) − MC(S ∪ {prev_best})
where `prev_best` is the candidate currently sitting at the top of the heap.
mg2 costs ONE extra MC call beyond mg1 (the baseline MC(S ∪ {prev_best}) is
shared and cached for the round).

If prev_best wins this round (committed as the next seed), then in the next
round mg1(v) for the new S' = S ∪ {prev_best} is exactly the cached mg2(v) —
no recomputation needed for any v that already has a fresh mg2.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from influence_max.agents.base import BaseAgent
from influence_max.config import AgentConfig
from influence_max.environment import State, Transition
from influence_max.graph import GraphData
from influence_max.ic_model import MCResult, MonteCarloEstimator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Heap entry shared by CELF and CELF++                                        #
# --------------------------------------------------------------------------- #


@dataclass(order=True)
class _HeapEntry:
    """Min-heap entry — we negate the gain to simulate a max-heap."""
    neg_mg: float                                    # primary sort key
    node: int = field(compare=False)
    last_eval_iter: int = field(compare=False)


# --------------------------------------------------------------------------- #
# CELF                                                                        #
# --------------------------------------------------------------------------- #


class CELFAgent(BaseAgent):
    """
    Lazy greedy with priority queue (Cost-Effective Lazy Forward).

    Identical *output* to GreedyMCAgent but with O(small) MC evaluations
    per seed selected instead of O(N).
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
        self._heap: List[_HeapEntry] = []
        self._initialized: bool = False
        # Cached spread of the current seed set — populated from env's MC result
        self._current_spread: float = 0.0
        # Diagnostic: number of MC evaluations performed this episode
        self._mc_eval_count: int = 0

    # ------------------------------------------------------------------ #
    # Episode lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset_episode(self) -> None:
        self._heap = []
        self._initialized = False
        self._current_spread = 0.0
        self._mc_eval_count = 0

    def end_episode(self, total_reward: float, episode_transitions: List[Transition]) -> None:
        naive_evals = self.graph.n_nodes * len(episode_transitions)
        if naive_evals > 0:
            speedup = naive_evals / max(1, self._mc_eval_count)
            logger.info(
                "CELF episode done: %d MC evaluations vs %d naive (%.1fx speedup)",
                self._mc_eval_count, naive_evals, speedup,
            )

    # ------------------------------------------------------------------ #
    # Seeding the heap on the first call                                   #
    # ------------------------------------------------------------------ #

    def _initialize_heap(self, valid_actions: List[int]) -> None:
        """Compute mg of every candidate against the empty seed set."""
        logger.info("CELF: seeding heap with %d candidates...", len(valid_actions))
        rollouts = self.config.mc_rollouts

        for v in valid_actions:
            result = self.mc.estimate(frozenset([v]), n_simulations=rollouts)
            mg = result.mean_spread
            self._mc_eval_count += 1
            heapq.heappush(self._heap, _HeapEntry(neg_mg=-mg, node=v, last_eval_iter=0))

        self._initialized = True
        logger.info("CELF: heap seeded with %d entries.", len(self._heap))

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        current_mc_result: Optional[MCResult] = None,
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")

        # Pick up the current spread for free from the env's MC result
        if current_mc_result is not None:
            self._current_spread = current_mc_result.mean_spread

        if not self._initialized:
            self._initialize_heap(valid_actions)

        rollouts = self.config.mc_rollouts
        seed_set = state.seed_set
        cur_iter = len(seed_set)
        valid_set = set(valid_actions)

        while True:
            entry = heapq.heappop(self._heap)
            v = entry.node

            # Skip nodes no longer valid (already seeded / activated)
            if v not in valid_set:
                continue

            if entry.last_eval_iter == cur_iter:
                # Top of heap has fresh mg w.r.t. current seed set →
                # submodularity guarantees this is the global argmax
                self.increment_step()
                logger.debug(
                    "CELF picked node=%d  mg=%.3f  (mc_evals=%d)",
                    v, -entry.neg_mg, self._mc_eval_count,
                )
                return v

            # Stale gain — recompute against the current seed set
            new_seed_set = seed_set | frozenset([v])
            result = self.mc.estimate(new_seed_set, n_simulations=rollouts)
            mg = result.mean_spread - self._current_spread
            self._mc_eval_count += 1

            heapq.heappush(self._heap, _HeapEntry(
                neg_mg=-mg, node=v, last_eval_iter=cur_iter,
            ))

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        new_spread = transition.info.get("mean_spread")
        if new_spread is not None:
            self._current_spread = float(new_spread)
        return None


# --------------------------------------------------------------------------- #
# CELF++                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class _PPEntry:
    """Per-candidate cache for CELF++."""
    mg1: float = 0.0                # marginal gain w.r.t. current S
    mg2: float = 0.0                # marginal gain w.r.t. S ∪ {prev_best}
    prev_best: Optional[int] = None # the prev_best mg2 was computed against
    flag: int = 0                   # last iteration mg1 was computed (== |S|)


class CELFppAgent(BaseAgent):
    """
    CELF with the prev-best look-ahead optimisation (Goyal et al. 2011).

    For each candidate v we cache (mg1, mg2, prev_best, flag).  When the
    candidate sitting at the top of the heap (`prev_best`) is committed as a
    seed, every cached mg2 whose `prev_best` matches becomes the candidate's
    new mg1 for free — saving one MC eval per such candidate.

    Implementation note: mg2 is only meaningful when prev_best has actually
    been committed; we identify "prev_best was committed" by comparing the
    seed set at evaluation time vs at consumption time.
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

        self._cache: Dict[int, _PPEntry] = {}
        self._heap: List[_HeapEntry] = []
        self._initialized: bool = False
        self._current_spread: float = 0.0
        # MC(S ∪ {prev_best}) cached once per round, used by every mg2 calculation
        self._prev_best_baseline: Optional[float] = None
        self._round_prev_best: Optional[int] = None
        self._mc_eval_count: int = 0
        # The node committed in the previous round (becomes "the new prev_best")
        self._last_committed: Optional[int] = None

    def reset_episode(self) -> None:
        self._cache = {}
        self._heap = []
        self._initialized = False
        self._current_spread = 0.0
        self._prev_best_baseline = None
        self._round_prev_best = None
        self._mc_eval_count = 0
        self._last_committed = None

    def end_episode(self, total_reward: float, episode_transitions: List[Transition]) -> None:
        naive_evals = self.graph.n_nodes * len(episode_transitions)
        if naive_evals > 0:
            speedup = naive_evals / max(1, self._mc_eval_count)
            logger.info(
                "CELF++ episode done: %d MC evaluations vs %d naive (%.1fx speedup)",
                self._mc_eval_count, naive_evals, speedup,
            )

    def _initialize(self, valid_actions: List[int]) -> None:
        logger.info("CELF++: seeding cache with %d candidates...", len(valid_actions))
        rollouts = self.config.mc_rollouts
        for v in valid_actions:
            result = self.mc.estimate(frozenset([v]), n_simulations=rollouts)
            mg = result.mean_spread
            self._mc_eval_count += 1
            self._cache[v] = _PPEntry(mg1=mg, mg2=0.0, prev_best=None, flag=0)
            heapq.heappush(self._heap, _HeapEntry(neg_mg=-mg, node=v, last_eval_iter=0))
        self._initialized = True

    def _peek_top_valid(self, valid_set: set, exclude: int) -> Optional[int]:
        """Find the node at the top of the heap that is still valid (best alternative to `exclude`)."""
        for entry in self._heap:
            if entry.node != exclude and entry.node in valid_set:
                return entry.node
        return None

    def select_action(
        self,
        state: State,
        valid_actions: List[int],
        current_mc_result: Optional[MCResult] = None,
        **kwargs: Any,
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")

        if current_mc_result is not None:
            self._current_spread = current_mc_result.mean_spread

        if not self._initialized:
            self._initialize(valid_actions)

        rollouts = self.config.mc_rollouts
        seed_set = state.seed_set
        cur_iter = len(seed_set)
        valid_set = set(valid_actions)

        # Reset the per-round prev_best baseline cache
        self._round_prev_best = None
        self._prev_best_baseline = None

        while True:
            entry = heapq.heappop(self._heap)
            v = entry.node

            if v not in valid_set:
                continue

            cached = self._cache.get(v)
            if cached is None:
                # Defensive: cold-start a candidate that wasn't in the initial set
                result = self.mc.estimate(seed_set | frozenset([v]), n_simulations=rollouts)
                mg = result.mean_spread - self._current_spread
                self._mc_eval_count += 1
                self._cache[v] = _PPEntry(mg1=mg, mg2=0.0, prev_best=None, flag=cur_iter)
                heapq.heappush(self._heap, _HeapEntry(
                    neg_mg=-mg, node=v, last_eval_iter=cur_iter,
                ))
                continue

            if cached.flag == cur_iter:
                # Fresh mg1 → committed
                self.increment_step()
                self._last_committed = v
                logger.debug(
                    "CELF++ picked node=%d  mg=%.3f  (mc_evals=%d)",
                    v, cached.mg1, self._mc_eval_count,
                )
                return v

            # CELF++ shortcut: mg2 was computed against the candidate that was
            # just committed → reuse it as the new mg1 for free.
            if (cached.prev_best is not None
                    and cached.prev_best == self._last_committed):
                cached.mg1 = cached.mg2
                cached.flag = cur_iter
                cached.prev_best = None
                cached.mg2 = 0.0
                heapq.heappush(self._heap, _HeapEntry(
                    neg_mg=-cached.mg1, node=v, last_eval_iter=cur_iter,
                ))
                continue

            # Stale → recompute mg1, opportunistically also compute mg2
            new_seed_set = seed_set | frozenset([v])
            result_v = self.mc.estimate(new_seed_set, n_simulations=rollouts)
            mg1 = result_v.mean_spread - self._current_spread
            self._mc_eval_count += 1

            # Look-ahead mg2 against the current top-of-heap (= prev_best for round)
            top = self._peek_top_valid(valid_set, exclude=v)
            mg2 = 0.0
            prev_best = None
            if top is not None:
                # Compute MC(S ∪ {top}) once per round and cache it
                if self._round_prev_best != top:
                    base_result = self.mc.estimate(seed_set | frozenset([top]),
                                                   n_simulations=rollouts)
                    self._prev_best_baseline = base_result.mean_spread
                    self._round_prev_best = top
                    self._mc_eval_count += 1
                # mg2(v) = MC(S ∪ {top, v}) − MC(S ∪ {top})
                combined = self.mc.estimate(seed_set | frozenset([top, v]),
                                            n_simulations=rollouts)
                mg2 = combined.mean_spread - self._prev_best_baseline
                self._mc_eval_count += 1
                prev_best = top

            cached.mg1 = mg1
            cached.mg2 = mg2
            cached.prev_best = prev_best
            cached.flag = cur_iter
            heapq.heappush(self._heap, _HeapEntry(
                neg_mg=-mg1, node=v, last_eval_iter=cur_iter,
            ))

    def update(self, transition: Transition) -> Optional[Dict[str, float]]:
        new_spread = transition.info.get("mean_spread")
        if new_spread is not None:
            self._current_spread = float(new_spread)
        return None
