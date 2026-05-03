"""
train_rl.py – CLI entry point for the influence maximization RL pipeline.

Runs one or more episodes of seed selection using the chosen agent strategy,
logs per-step details, and saves results to results/<run_id>.json.

Usage
-----
# Greedy Monte Carlo (default)
python train_rl.py --budget 10 --mc-sims 100 --agent-mc-rollouts 50

# Random baseline
python train_rl.py --agent random --budget 10

# Degree heuristic baseline
python train_rl.py --agent degree --budget 10

# Override everything via a JSON config file
python train_rl.py --config my_config.json

Run `python train_rl.py --help` for all options.

Standardization notes for future algorithms
-------------------------------------------
To add Q-learning or DQN:
  1. Implement BaseAgent in influence_max/agents/<algo>.py
  2. Register the strategy string in agents/monte_carlo.py::build_agent()
  3. Add any new AgentConfig fields in influence_max/config.py
  4. The episode loop below does NOT change.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from influence_max.agents.monte_carlo import build_agent
from influence_max.config import (
    AgentConfig,
    EnvConfig,
    GraphConfig,
    ICConfig,
    TrainConfig,
    resolve_device,
)
from influence_max.environment import InfluenceMaxEnv, Transition
from influence_max.graph import load_graph
from influence_max.ic_model import ICDiffusion, MonteCarloEstimator


# --------------------------------------------------------------------------- #
# Logging setup                                                                #
# --------------------------------------------------------------------------- #


def setup_logging(level: str) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Seeding                                                                      #
# --------------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Episode runner                                                                #
# --------------------------------------------------------------------------- #


def run_episode(
    env: InfluenceMaxEnv,
    agent,
    episode_idx: int,
) -> Dict[str, Any]:
    """
    Run a single episode and return a structured result dict.

    The episode loop:
      1. reset env  -> initial state, info
      2. while not done:
           a. compute valid_actions from action_mask
           b. agent.select_action(state, valid_actions, current_mc_result)
           c. env.step(action) -> next_state, reward, done, info
           d. agent.update(transition)  [no-op for planning agents]
      3. agent.end_episode(total_reward, transitions)
    """
    agent.reset_episode()
    state, info = env.reset()

    total_reward = 0.0
    steps: List[Dict[str, Any]] = []
    done = False

    logger.info("=" * 60)
    logger.info("Episode %d start  budget=%d", episode_idx, state.budget)
    logger.info("=" * 60)

    step_idx = 0
    while not done:
        union_activated = info["union_activated"]
        current_mc = info["mc_result"]

        valid_actions = env.valid_actions(state, union_activated)
        if not valid_actions:
            logger.warning(
                "No valid actions at step %d – terminating episode.", step_idx
            )
            break

        action = agent.select_action(
            state,
            valid_actions,
            current_mc_result=current_mc,
        )

        next_state, reward, done, info = env.step(action)

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
        )
        agent.update(transition)

        total_reward += reward
        step_record = {
            "step": step_idx,
            "action": action,
            "reward": round(reward, 4),
            "mean_spread": round(info["mean_spread"], 4),
            "std_spread": round(info["std_spread"], 4),
            "seed_set": sorted(next_state.seed_set),
            "budget_remaining": next_state.budget,
            "n_activated_union": len(info["union_activated"]),
        }
        steps.append(step_record)

        logger.info(
            "  Step %2d | action=%-6d | reward=%7.3f | spread=%.2f±%.2f | "
            "seeds=%s | activated=%d",
            step_idx,
            action,
            reward,
            info["mean_spread"],
            info["std_spread"],
            sorted(next_state.seed_set),
            len(info["union_activated"]),
        )

        state = next_state
        step_idx += 1

    agent.end_episode(total_reward, env.episode_transitions)

    final_seed_set = sorted(state.seed_set)
    final_spread = info.get("mean_spread", 0.0)

    logger.info("-" * 60)
    logger.info(
        "Episode %d done | total_reward=%.4f | final_spread=%.2f | seeds=%s",
        episode_idx,
        total_reward,
        final_spread,
        final_seed_set,
    )

    return {
        "episode": episode_idx,
        "total_reward": round(total_reward, 4),
        "final_spread": round(final_spread, 4),
        "final_seed_set": final_seed_set,
        "n_steps": step_idx,
        "steps": steps,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Influence Maximization RL – train / evaluate an agent."
    )

    # Config file (overrides all other flags if provided)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON TrainConfig file. Overrides all other flags.",
    )

    # Graph
    parser.add_argument("--data-path", type=str, default="data/p2p-Gnutella08.txt")

    # IC model
    parser.add_argument(
        "--embeddings",
        type=str,
        default="node_embeddings.pt",
        help="Path to saved node embeddings (.pt file).",
    )
    parser.add_argument(
        "--mc-sims",
        type=int,
        default=100,
        help="MC simulations for reward computation in the env.",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=1e-4,
        help="Min edge probability to retain in IC cache.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help=(
            "Compute device for embedding dot-products. "
            "'auto' selects MPS (Apple GPU) > CUDA > CPU automatically."
        ),
    )

    # Environment
    parser.add_argument(
        "--budget", type=int, default=10, help="Seed budget per episode."
    )

    # Agent
    parser.add_argument(
        "--agent",
        type=str,
        default="greedy_mc",
        choices=[
            "greedy_mc", "random", "degree",
            "dqn", "sarsa", "linucb",
            "celf", "celfpp",
            "imm",
            "pagerank", "kshell", "betweenness", "degree_discount",
            "louvain",
            "s2v_dqn",
        ],
        help="Agent strategy.",
    )
    parser.add_argument(
        "--agent-mc-rollouts",
        type=int,
        default=50,
        help="MC rollouts used by greedy MC agents (GreedyMC / CELF / CELF++ / Louvain).",
    )

    # Learning hyperparameters (DQN / SARSA)
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (DQN/SARSA)."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor (DQN/SARSA)."
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=500,
        help="Number of agent steps over which epsilon decays.",
    )

    # DQN-specific
    parser.add_argument("--replay-buffer-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update-freq", type=int, default=10)

    # LinUCB-specific
    parser.add_argument(
        "--linucb-alpha",
        type=float,
        default=1.0,
        help="UCB exploration coefficient (LinUCB).",
    )
    parser.add_argument(
        "--linucb-lambda",
        type=float,
        default=1.0,
        help="Ridge regularisation strength (LinUCB).",
    )

    # IMM-specific
    parser.add_argument("--imm-epsilon", type=float, default=0.1,
                        help="IMM approximation error parameter.")
    parser.add_argument("--imm-delta", type=float, default=0.0,
                        help="IMM failure prob (0 → defaults to 1/N).")
    parser.add_argument("--imm-theta-max", type=int, default=200_000,
                        help="Cap on IMM RR-set count (runtime safety).")

    # Centrality-specific
    parser.add_argument("--pagerank-alpha", type=float, default=0.85,
                        help="PageRank damping factor.")
    parser.add_argument("--betweenness-n-samples", type=int, default=1000,
                        help="Source-node samples for approximate betweenness "
                             "(only used when n_nodes > 5000).")
    parser.add_argument("--dd-propagation-p", type=float, default=0.0,
                        help="DegreeDiscount propagation prob "
                             "(0 → use mean of IC edge probabilities).")

    # Community
    parser.add_argument("--community-resolution", type=float, default=1.0,
                        help="Louvain resolution (>1 → more, smaller communities).")

    # S2V-DQN-specific
    parser.add_argument("--s2v-t-iters", type=int, default=4,
                        help="Number of Structure2Vec message-passing iterations.")
    parser.add_argument("--s2v-hidden-dim", type=int, default=64,
                        help="Hidden / embedding dim for S2V-DQN.")

    # Training
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Build config                                                         #
    # ------------------------------------------------------------------ #
    if args.config:
        cfg = TrainConfig.from_json(args.config)
    else:
        cfg = TrainConfig(
            graph=GraphConfig(data_path=args.data_path),
            ic=ICConfig(
                embeddings_path=args.embeddings,
                n_simulations=args.mc_sims,
                prob_threshold=args.prob_threshold,
                seed=args.seed,
                device=args.device,
            ),
            env=EnvConfig(budget=args.budget),
            agent=AgentConfig(
                strategy=args.agent,
                mc_rollouts=args.agent_mc_rollouts,
                learning_rate=args.lr,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay=args.epsilon_decay,
                replay_buffer_size=args.replay_buffer_size,
                batch_size=args.batch_size,
                target_update_freq=args.target_update_freq,
                linucb_alpha=args.linucb_alpha,
                linucb_lambda=args.linucb_lambda,
                imm_epsilon=args.imm_epsilon,
                imm_delta=args.imm_delta,
                imm_theta_max=args.imm_theta_max,
                pagerank_alpha=args.pagerank_alpha,
                betweenness_n_samples=args.betweenness_n_samples,
                dd_propagation_p=args.dd_propagation_p,
                community_resolution=args.community_resolution,
                s2v_t_iters=args.s2v_t_iters,
                s2v_hidden_dim=args.s2v_hidden_dim,
                embeddings_path=args.embeddings,
            ),
            n_episodes=args.episodes,
            results_dir=args.results_dir,
            log_level=args.log_level,
            seed=args.seed,
        )

    setup_logging(cfg.log_level)
    seed_everything(cfg.seed)

    resolved = resolve_device(cfg.ic.device)
    logger.info("Configuration:\n%s", json.dumps(cfg.to_dict(), indent=2))
    logger.info("Compute device: %s", resolved)

    # ------------------------------------------------------------------ #
    # Build components                                                     #
    # ------------------------------------------------------------------ #
    logger.info("Loading graph from %s ...", cfg.graph.data_path)
    graph = load_graph(cfg.graph.data_path)

    logger.info("Building IC diffusion model ...")
    ic = ICDiffusion(graph, cfg.ic)
    mc = MonteCarloEstimator(ic, cfg.ic)

    env = InfluenceMaxEnv(graph, ic, mc, cfg.env)

    agent = build_agent(
        cfg.agent,
        mc,
        graph,
        budget=cfg.env.budget,
        device=cfg.ic.device,
        seed=cfg.seed,
        ic=ic,
    )
    logger.info("Agent: %s", agent)

    # ------------------------------------------------------------------ #
    # Episode loop                                                         #
    # ------------------------------------------------------------------ #
    all_results: List[Dict[str, Any]] = []
    t_start = time.time()

    for ep in range(cfg.n_episodes):
        ep_result = run_episode(env, agent, episode_idx=ep)
        all_results.append(ep_result)

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    spreads = [r["final_spread"] for r in all_results]
    rewards = [r["total_reward"] for r in all_results]
    logger.info(
        "\n%s\nSummary over %d episode(s):\n"
        "  Spread  – mean=%.2f  std=%.2f  min=%.2f  max=%.2f\n"
        "  Reward  – mean=%.4f  std=%.4f\n"
        "  Time    – %.1fs",
        "=" * 60,
        cfg.n_episodes,
        float(np.mean(spreads)),
        float(np.std(spreads)),
        float(np.min(spreads)),
        float(np.max(spreads)),
        float(np.mean(rewards)),
        float(np.std(rewards)),
        elapsed,
    )

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{cfg.agent.strategy}"
    out_path = results_dir / f"{run_id}.json"

    output = {
        "run_id": run_id,
        "config": cfg.to_dict(),
        "elapsed_seconds": round(elapsed, 2),
        "summary": {
            "n_episodes": cfg.n_episodes,
            "mean_spread": round(float(np.mean(spreads)), 4),
            "std_spread": round(float(np.std(spreads)), 4),
            "mean_reward": round(float(np.mean(rewards)), 4),
        },
        "episodes": all_results,
    }

    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
