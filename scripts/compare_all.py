"""
Comprehensive benchmark runner — runs every implemented agent under identical
conditions and writes a unified results JSON + a markdown report.

Usage
-----
# Default sweep: all 14 agents, budget=10, mc-sims=100 (matches
# the figures in docs/RESULTS_ANALYSIS.md)
python scripts/compare_all.py

# Quick sanity sweep
python scripts/compare_all.py --budget 5 --mc-sims 30 --agent-mc-rollouts 20

# Subset
python scripts/compare_all.py --include greedy_mc celf imm pagerank dqn

Results
-------
- results/benchmark_<timestamp>.json   — full per-agent traces, configs, timings
- docs/COMPREHENSIVE_BENCHMARK.md      — auto-generated markdown report

Note: GreedyMC, learning agents (DQN, SARSA, S2V-DQN) and Betweenness can be
slow on the 6301-node graph.  Use `--exclude greedy_mc dqn sarsa s2v_dqn`
for a quick sweep, or `--budget 3` to shrink runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running from project root: `python scripts/compare_all.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Agent registry — (strategy_id, family, label, extra_kwargs_for_AgentConfig) #
# --------------------------------------------------------------------------- #


AGENT_REGISTRY: List[Tuple[str, str, str, Dict[str, Any]]] = [
    # Baselines
    ("random",          "Baseline",        "Random",          {}),
    ("degree",          "Baseline",        "Degree",          {}),
    # Simulation-based greedy
    ("greedy_mc",       "Sim greedy",      "GreedyMC",        {}),
    ("celf",            "Sim greedy",      "CELF",            {}),
    ("celfpp",          "Sim greedy",      "CELF++",          {}),
    # Reverse Influence Sampling
    ("imm",             "RIS",             "IMM",             {}),
    # Centrality / heuristics
    ("pagerank",        "Centrality",      "PageRank",        {}),
    ("kshell",          "Centrality",      "k-shell",         {}),
    ("betweenness",     "Centrality",      "Betweenness",     {}),
    ("degree_discount", "Centrality",      "DegreeDiscount",  {}),
    # Community-based
    ("louvain",         "Community",       "Louvain+greedy",  {}),
    # Learning / DL
    ("dqn",             "Learning",        "DQN",             {}),
    ("sarsa",           "Learning",        "SARSA",           {}),
    ("linucb",          "Learning",        "LinUCB",          {}),
    ("s2v_dqn",         "Learning DL",     "S2V-DQN",         {}),
]


# --------------------------------------------------------------------------- #
# Episode runner (replicates train_rl.py semantics)                            #
# --------------------------------------------------------------------------- #


def run_episode(env: InfluenceMaxEnv, agent, episode_idx: int) -> Dict[str, Any]:
    agent.reset_episode()
    state, info = env.reset()

    total_reward = 0.0
    steps: List[Dict[str, Any]] = []
    done = False
    step_idx = 0

    while not done:
        union_activated = info["union_activated"]
        current_mc = info["mc_result"]
        valid_actions = env.valid_actions(state, union_activated)
        if not valid_actions:
            break

        action = agent.select_action(
            state, valid_actions, current_mc_result=current_mc,
        )
        next_state, reward, done, info = env.step(action)

        transition = Transition(
            state=state, action=action, reward=reward,
            next_state=next_state, done=done, info=info,
        )
        agent.update(transition)

        total_reward += reward
        steps.append({
            "step": step_idx,
            "action": action,
            "reward": round(reward, 4),
            "mean_spread": round(info["mean_spread"], 4),
            "seed_set": sorted(next_state.seed_set),
            "n_activated_union": len(info["union_activated"]),
        })
        state = next_state
        step_idx += 1

    agent.end_episode(total_reward, env.episode_transitions)

    return {
        "episode": episode_idx,
        "total_reward": round(total_reward, 4),
        "final_spread": round(info.get("mean_spread", 0.0), 4),
        "final_seed_set": sorted(state.seed_set),
        "n_steps": step_idx,
        "steps": steps,
    }


# --------------------------------------------------------------------------- #
# Per-agent benchmark                                                          #
# --------------------------------------------------------------------------- #


def benchmark_one(
    strategy: str,
    family: str,
    label: str,
    extra_cfg: Dict[str, Any],
    args: argparse.Namespace,
    graph,
    ic: ICDiffusion,
    mc: MonteCarloEstimator,
) -> Dict[str, Any]:
    """Run one agent for `args.episodes` episodes and collect its result."""
    logger.info("=" * 70)
    logger.info("Benchmarking %s  (%s)", label, family)
    logger.info("=" * 70)

    agent_cfg = AgentConfig(
        strategy=strategy,
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
        community_resolution=args.community_resolution,
        s2v_t_iters=args.s2v_t_iters,
        s2v_hidden_dim=args.s2v_hidden_dim,
        embeddings_path=args.embeddings,
        **extra_cfg,
    )
    env_cfg = EnvConfig(budget=args.budget)
    env = InfluenceMaxEnv(graph, ic, mc, env_cfg)

    try:
        agent = build_agent(
            agent_cfg, mc, graph,
            budget=args.budget, device=args.device, seed=args.seed, ic=ic,
        )
    except Exception as e:
        logger.error("Failed to build %s: %s", label, e)
        return {
            "strategy": strategy, "family": family, "label": label,
            "error": str(e), "elapsed_sec": 0.0,
        }

    t0 = time.time()
    episodes: List[Dict[str, Any]] = []
    try:
        for ep in range(args.episodes):
            ep_result = run_episode(env, agent, episode_idx=ep)
            episodes.append(ep_result)
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("%s crashed after %.1fs: %s\n%s",
                     label, elapsed, e, traceback.format_exc())
        return {
            "strategy": strategy, "family": family, "label": label,
            "error": str(e), "elapsed_sec": round(elapsed, 2),
            "episodes": episodes,
        }
    elapsed = time.time() - t0

    spreads = np.array([e["final_spread"] for e in episodes], dtype=float)
    rewards = np.array([e["total_reward"] for e in episodes], dtype=float)
    last_seed_set = episodes[-1]["final_seed_set"] if episodes else []

    summary = {
        "strategy": strategy,
        "family": family,
        "label": label,
        "elapsed_sec": round(elapsed, 2),
        "n_episodes": len(episodes),
        "mean_spread": round(float(spreads.mean()), 4),
        "std_spread": round(float(spreads.std()), 4),
        "min_spread": round(float(spreads.min()), 4),
        "max_spread": round(float(spreads.max()), 4),
        "mean_reward": round(float(rewards.mean()), 4),
        "last_seed_set": last_seed_set,
        "episodes": episodes,
    }
    logger.info(
        "  %s done | spread=%.2f±%.2f | time=%.1fs | seeds=%s",
        label, summary["mean_spread"], summary["std_spread"], elapsed, last_seed_set,
    )
    return summary


# --------------------------------------------------------------------------- #
# Markdown report generator                                                    #
# --------------------------------------------------------------------------- #


def generate_markdown_report(
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Render the per-agent results into a comparison report."""
    lines: List[str] = []
    lines.append("# Comprehensive Influence Maximization Benchmark")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("Auto-generated from `scripts/compare_all.py`. All agents run under "
                 "identical conditions:")
    lines.append("")
    lines.append(f"- **Dataset**: `{args.data_path}` (p2p-Gnutella08)")
    lines.append(f"- **Budget (k)**: {args.budget}")
    lines.append(f"- **MC simulations (env reward)**: {args.mc_sims}")
    lines.append(f"- **MC rollouts (greedy agents)**: {args.agent_mc_rollouts}")
    lines.append(f"- **Episodes per agent**: {args.episodes}")
    lines.append(f"- **Seed**: {args.seed}")
    lines.append(f"- **Device**: {args.device}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Top-line comparison table                                            #
    # ------------------------------------------------------------------ #
    lines.append("## Headline comparison")
    lines.append("")
    lines.append("| Family | Algorithm | Spread (mean ± std) | Time (s) | Final seeds |")
    lines.append("| ------ | --------- | ------------------- | -------- | ----------- |")
    # Sort: by family, then by mean spread descending
    family_order = ["Baseline", "Sim greedy", "RIS", "Centrality",
                    "Community", "Learning", "Learning DL"]
    def sort_key(r: Dict[str, Any]) -> Tuple[int, float]:
        fam_idx = family_order.index(r["family"]) if r["family"] in family_order else 99
        return (fam_idx, -float(r.get("mean_spread", 0.0)))
    for r in sorted(results, key=sort_key):
        if "error" in r:
            spread_str = f"_error: {r['error'][:40]}_"
            seeds_str = "—"
        else:
            spread_str = f"{r['mean_spread']:.2f} ± {r['std_spread']:.2f}"
            seeds_str = str(r.get("last_seed_set", []))
            if len(seeds_str) > 60:
                seeds_str = seeds_str[:60] + "…"
        lines.append(
            f"| {r['family']} | **{r['label']}** | {spread_str} | "
            f"{r['elapsed_sec']:.1f} | `{seeds_str}` |"
        )
    lines.append("")

    # ------------------------------------------------------------------ #
    # Per-family discussion                                                #
    # ------------------------------------------------------------------ #
    lines.append("---")
    lines.append("")
    lines.append("## Per-family analysis")
    lines.append("")

    fam_takeaways: Dict[str, str] = {
        "Baseline": (
            "Random and pure Degree provide the lower / upper sanity bounds. "
            "Any algorithm that fails to beat Degree on this graph is not "
            "exploiting any structure beyond local connectivity."
        ),
        "Sim greedy": (
            "GreedyMC, CELF and CELF++ produce identical seed sets (submodular "
            "guarantee) but differ massively in runtime: CELF typically "
            "100×–1000× faster than naive greedy, and CELF++ ~2× faster than CELF. "
            "These are the gold-standard accuracy baselines for IM."
        ),
        "RIS": (
            "IMM trades a tiny amount of spread for dramatic runtime gains. "
            "On large graphs (≫ 10⁵ nodes) it is typically the only tractable "
            "algorithm with theoretical guarantees ((1 − 1/e − ε) approximation)."
        ),
        "Centrality": (
            "Centrality heuristics are blazingly fast (sub-second). Their relative "
            "performance reflects how well static graph structure correlates with "
            "expected spread under our embedding-based IC model. "
            "DegreeDiscount is the only state-aware variant — the others ignore "
            "diffusion overlap entirely."
        ),
        "Community": (
            "Louvain reduces the candidate space by partitioning the graph and "
            "running greedy within each community. Quality is bounded by partition "
            "quality; on dense P2P graphs like Gnutella, large communities make "
            "this gain over plain greedy modest."
        ),
        "Learning": (
            "DQN, SARSA, and LinUCB use the pretrained GNN embeddings. They learn "
            "a policy (DQN/SARSA) or contextual ranking (LinUCB) over the embedding "
            "space. Performance depends heavily on the quality of the embeddings "
            "and the exploration schedule."
        ),
        "Learning DL": (
            "S2V-DQN trains its own node embeddings end-to-end *together with* the "
            "Q-function via Structure2Vec message passing. This bypasses the GNN "
            "pretraining step and makes the encoder task-aware. In exchange, the "
            "training signal is much noisier and slower to converge than DQN with "
            "pretrained embeddings."
        ),
    }
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r)

    for fam in family_order:
        if fam not in by_family:
            continue
        lines.append(f"### {fam}")
        lines.append("")
        lines.append(fam_takeaways.get(fam, ""))
        lines.append("")
        lines.append("| Algorithm | Spread (mean ± std) | Time (s) |")
        lines.append("| --------- | ------------------- | -------- |")
        for r in sorted(by_family[fam],
                        key=lambda x: -float(x.get("mean_spread", 0.0))):
            if "error" in r:
                lines.append(
                    f"| **{r['label']}** | _error: {r['error'][:40]}_ | "
                    f"{r['elapsed_sec']:.1f} |"
                )
            else:
                lines.append(
                    f"| **{r['label']}** | "
                    f"{r['mean_spread']:.2f} ± {r['std_spread']:.2f} | "
                    f"{r['elapsed_sec']:.1f} |"
                )
        lines.append("")

    # ------------------------------------------------------------------ #
    # Per-algorithm step trace (last episode, top 8 by spread)             #
    # ------------------------------------------------------------------ #
    lines.append("---")
    lines.append("")
    lines.append("## Per-algorithm seeding trace (last episode)")
    lines.append("")
    lines.append("Step-by-step seed selection for each agent. Reveals diversity of "
                 "selected hubs vs. submodular gain captured.")
    lines.append("")

    for r in sorted(results,
                    key=lambda x: -float(x.get("mean_spread", 0.0)))[:12]:
        if "error" in r or not r.get("episodes"):
            continue
        lines.append(f"### {r['label']} ({r['family']})")
        lines.append("")
        lines.append("| Step | Action | Reward | Cumulative spread |")
        lines.append("| ---- | ------ | ------ | ----------------- |")
        last_ep = r["episodes"][-1]
        for s in last_ep["steps"]:
            lines.append(
                f"| {s['step']} | {s['action']} | {s['reward']:+.2f} | "
                f"{s['mean_spread']:.2f} |"
            )
        lines.append("")

    # ------------------------------------------------------------------ #
    # Takeaways                                                            #
    # ------------------------------------------------------------------ #
    lines.append("---")
    lines.append("")
    lines.append("## Key takeaways")
    lines.append("")
    successful = [r for r in results if "error" not in r]
    if successful:
        best = max(successful, key=lambda x: float(x["mean_spread"]))
        fastest = min(successful, key=lambda x: float(x["elapsed_sec"]))
        baseline = next((r for r in successful if r["strategy"] == "degree"), None)

        lines.append(f"- **Best spread**: **{best['label']}** with "
                     f"{best['mean_spread']:.2f} (family: {best['family']}, "
                     f"{best['elapsed_sec']:.1f}s).")
        lines.append(f"- **Fastest**: **{fastest['label']}** at "
                     f"{fastest['elapsed_sec']:.2f}s "
                     f"(spread: {fastest['mean_spread']:.2f}).")

        if baseline is not None:
            beats_baseline = [
                r for r in successful
                if r["mean_spread"] > baseline["mean_spread"] + 1e-3
            ]
            lines.append(f"- **Beats Degree baseline ({baseline['mean_spread']:.2f})**: "
                         f"{len(beats_baseline)} / {len(successful)} agents — "
                         f"{', '.join(r['label'] for r in beats_baseline)}.")

        lines.append("")
        lines.append("- **Family ranking** (by mean spread, top of family):")
        for fam in family_order:
            if fam not in by_family:
                continue
            ok = [r for r in by_family[fam] if "error" not in r]
            if not ok:
                continue
            top = max(ok, key=lambda x: float(x["mean_spread"]))
            lines.append(f"  - {fam}: {top['label']} ({top['mean_spread']:.2f})")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_See `results/benchmark_*.json` for raw per-step traces._")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Markdown report written to %s", output_path)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def setup_logging(level: str) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s – %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Comprehensive IM benchmark: run every agent and produce a unified report."
    )
    # Graph / IC
    p.add_argument("--data-path", default="data/p2p-Gnutella08.txt")
    p.add_argument("--embeddings", default="node_embeddings.pt")
    p.add_argument("--mc-sims", type=int, default=100,
                   help="MC simulations for env reward computation.")
    p.add_argument("--prob-threshold", type=float, default=1e-4)
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--budget", type=int, default=10)

    # Agents
    p.add_argument("--include", nargs="*", default=None,
                   help="Subset of strategy ids to benchmark "
                        "(default: all in registry).")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="Strategy ids to skip.")
    p.add_argument("--agent-mc-rollouts", type=int, default=50)

    # Learning hyperparams
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=50)
    p.add_argument("--replay-buffer-size", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-update-freq", type=int, default=10)
    p.add_argument("--linucb-alpha", type=float, default=1.0)
    p.add_argument("--linucb-lambda", type=float, default=1.0)
    p.add_argument("--imm-epsilon", type=float, default=0.1)
    p.add_argument("--imm-delta", type=float, default=0.0)
    p.add_argument("--imm-theta-max", type=int, default=200_000)
    p.add_argument("--pagerank-alpha", type=float, default=0.85)
    p.add_argument("--betweenness-n-samples", type=int, default=500)
    p.add_argument("--community-resolution", type=float, default=1.0)
    p.add_argument("--s2v-t-iters", type=int, default=4)
    p.add_argument("--s2v-hidden-dim", type=int, default=64)

    # Run
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--report-path", default="docs/COMPREHENSIVE_BENCHMARK.md")
    p.add_argument("--log-level", default="INFO")

    args = p.parse_args()

    setup_logging(args.log_level)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    logger.info("Loading graph + IC ...")
    graph = load_graph(args.data_path)
    ic_cfg = ICConfig(
        embeddings_path=args.embeddings,
        n_simulations=args.mc_sims,
        prob_threshold=args.prob_threshold,
        seed=args.seed,
        device=args.device,
    )
    ic = ICDiffusion(graph, ic_cfg)
    mc = MonteCarloEstimator(ic, ic_cfg)

    logger.info("Compute device: %s", resolve_device(args.device))

    # Filter the registry
    include = set(args.include) if args.include else None
    exclude = set(args.exclude) if args.exclude else set()
    plan = [
        e for e in AGENT_REGISTRY
        if (include is None or e[0] in include) and e[0] not in exclude
    ]
    logger.info("Will benchmark %d agents: %s",
                len(plan), [e[2] for e in plan])

    results: List[Dict[str, Any]] = []
    overall_t0 = time.time()
    for strategy, family, label, extra_cfg in plan:
        result = benchmark_one(
            strategy, family, label, extra_cfg, args, graph, ic, mc,
        )
        results.append(result)
    overall_elapsed = time.time() - overall_t0

    # ------------------------------------------------------------------ #
    # Save raw JSON                                                        #
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"benchmark_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "config": vars(args),
        "overall_elapsed_sec": round(overall_elapsed, 2),
        "results": results,
    }
    json_path.write_text(json.dumps(output, indent=2))
    logger.info("Raw results -> %s", json_path)

    # ------------------------------------------------------------------ #
    # Generate markdown report                                             #
    # ------------------------------------------------------------------ #
    report_path = Path(args.report_path)
    generate_markdown_report(results, args, report_path)

    # ------------------------------------------------------------------ #
    # Console summary                                                      #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 80)
    print("Benchmark complete")
    print("=" * 80)
    print(f"Total runtime: {overall_elapsed:.1f}s")
    print(f"JSON results : {json_path}")
    print(f"Report       : {report_path}")
    print()
    print(f"{'Family':<14} {'Algorithm':<18} {'Spread':>14} {'Time(s)':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: -float(x.get("mean_spread", -1))):
        if "error" in r:
            print(f"{r['family']:<14} {r['label']:<18} {'ERR':>14} "
                  f"{r['elapsed_sec']:>10.1f}")
        else:
            spread = f"{r['mean_spread']:.2f}±{r['std_spread']:.2f}"
            print(f"{r['family']:<14} {r['label']:<18} {spread:>14} "
                  f"{r['elapsed_sec']:>10.1f}")


if __name__ == "__main__":
    main()
