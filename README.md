# Influence Maximization via RL

Influence maximization on the **p2p-Gnutella08** graph (6,301 nodes, 20,777 directed edges) using a Reinforcement Learning pipeline built on learned GNN embeddings.

## How it works

**Embeddings** → `train_gnn.py` trains a GraphSAGE + GAE model to learn 128-dim node embeddings. Edge influence probabilities are derived as `P(u→v) = σ(Hᵤ · Hᵥ)` (best val AUC: 0.924).

**IC Diffusion** → `influence_max/ic_model.py` uses those probabilities to run Independent Cascade simulations. Monte Carlo averaging over N rollouts gives the expected spread.

**RL Environment** → State `S_t = (seed_set, budget)`. Activated nodes are computed on-demand from the IC model — not stored in state — and used to mask out invalid actions. Reward is the marginal influence gain:

```
r_t = |Activated(seed_t+1)| − |Activated(seed_t)|
```

**Agents** → All 14 strategies share the same `BaseAgent` interface and slot into the same env / runner unchanged:

| Agent                    | Family                  | Strategy                                                                    |
| ------------------------ | ----------------------- | --------------------------------------------------------------------------- |
| `RandomAgent`            | baseline                | uniform random over valid actions                                           |
| `DegreeHeuristicAgent`   | baseline                | pick highest out-degree unselected node                                     |
| `GreedyMCAgent`          | simulation greedy       | argmax marginal MC gain across all valid candidates                         |
| `CELFAgent`              | simulation greedy       | lazy heap over MC marginal gains (Leskovec et al. 2007)                     |
| `CELFppAgent`            | simulation greedy       | CELF + look-ahead `mg2` cache (Goyal et al. 2011)                           |
| `IMMAgent`               | reverse influence sampling | reverse-BFS RR-set sampling + greedy max-cover (Tang et al. 2015)        |
| `PageRankAgent`          | centrality              | top-k by PageRank on the directed graph                                     |
| `KShellAgent`            | centrality              | top-k by k-core decomposition (Kitsak et al. 2010)                          |
| `BetweennessAgent`       | centrality              | top-k by betweenness (exact or sampled)                                     |
| `DegreeDiscountAgent`    | centrality (state-aware)| `score(v) = d_v − 2·t_v − (d_v−t_v)·t_v·p` (Chen et al. 2009)               |
| `LouvainAgent`           | community               | Louvain partition + per-community greedy MC                                 |
| `DQNAgent`               | off-policy Q-learning   | MLP `Q(s,a)` + replay buffer + target net + ε-greedy (uses pretrained GNN)  |
| `SARSAAgent`             | on-policy TD(0)         | linear `Q(s,a) = w·φ(s,a)` + on-policy bootstrap (uses pretrained GNN)      |
| `LinUCBAgent`            | contextual bandit       | linear ridge regression + UCB exploration (uses pretrained GNN)             |
| `S2VDQNAgent`            | learning / DL           | Structure2Vec encoder jointly trained with DQN (Khalil et al. 2017)         |

- See [`docs/RL_AGENTS.md`](docs/RL_AGENTS.md) for the full algorithmic reference (formulas + per-episode flowcharts) for the original three RL agents (DQN / SARSA / LinUCB).
- See [`docs/RESULTS_ANALYSIS.md`](docs/RESULTS_ANALYSIS.md) for the original RL-agent learning curves, seed diversity, and takeaways.
- See [`docs/COMPREHENSIVE_BENCHMARK.md`](docs/COMPREHENSIVE_BENCHMARK.md) for the head-to-head comparison of all 14 agents under identical conditions, with per-family analysis tied back to the project's strength / limitation claims.

## Results summary (budget = 10, 100 MC sims, identical env / IC model)

The full head-to-head numbers live in [`docs/COMPREHENSIVE_BENCHMARK.md`](docs/COMPREHENSIVE_BENCHMARK.md). Top-line:

| Family            | Algorithm        | Spread       | Time      |
| ----------------- | ---------------- | -----------: | --------: |
| RIS               | **IMM**          | **4449.18**  | 5,265 s   |
| Sim greedy        | GreedyMC         | 4430.05      | 618 s     |
| Baseline          | Degree           | 4422.73      | 544 s     |
| Community         | Louvain + greedy | 4417.50      | 4,621 s   |
| Learning          | DQN (pretrained) | 4416.77      | 565 s     |
| Centrality        | DegreeDiscount   | 4413.44      | **7 s**   |
| Sim greedy        | CELF++           | 4408.85      | 14,622 s  |
| Learning          | SARSA            | 4393.43      | 427 s     |
| Centrality        | PageRank         | 4389.91      | 8 s       |
| Baseline          | Random           | 4379.14      | 4 s       |
| Centrality        | k-shell          | 4376.78      | 8 s       |
| Learning          | LinUCB           | 2233.22      | 1 s       |
| Learning DL       | S2V-DQN best ep  | 4391.34      | 155 s     |

**Key findings**: IMM is the strongest spread; DegreeDiscount is the best speed/quality trade-off (within 0.8 % of GreedyMC at 7 seconds); LinUCB and S2V-DQN both exhibit the failure modes the project plan predicted (no sequential modelling and training instability respectively).

### Greedy MC reference seed set (single deterministic run)

```
Seeds: [194, 516, 636, 994, 1207, 1462, 1649, 1857, 2835, 5410]
Spread: 4430.05 / 6301 nodes  (~70.3% of the network)
```

| Step | Seed added | Spread  | Marginal gain |
| ---- | ---------- | ------- | ------------- |
| 0    | 5410       | 4385.73 | +4385.73      |
| 1    | 1207       | 4387.46 | +1.73         |
| 2    | 636        | 4396.40 | +8.94         |
| 3    | 1857       | 4399.59 | +3.19         |
| 4    | 2835       | 4397.71 | −1.88         |
| 5    | 994        | 4413.29 | +15.58        |
| 6    | 194        | 4403.81 | −9.48         |
| 7    | 1649       | 4420.32 | +16.51        |
| 8    | 1462       | 4421.69 | +1.37         |
| 9    | 516        | 4430.05 | +8.36         |

## Project structure

```
influence_max/
  config.py        # GraphConfig, ICConfig, EnvConfig, AgentConfig (all 14 strategies)
  graph.py         # edge-list loader, adjacency dicts
  ic_model.py      # ICDiffusion + MonteCarloEstimator
  environment.py   # InfluenceMaxEnv, State, Transition
  agents/
    base.py        # BaseAgent (abstract)
    monte_carlo.py # GreedyMC, Random, DegreeHeuristic + build_agent factory
    celf.py        # CELF + CELF++ (lazy greedy, Leskovec '07 / Goyal '11)
    ris.py         # IMMAgent (Reverse Influence Sampling, Tang et al. '15)
    centrality.py  # PageRank / k-shell / Betweenness / DegreeDiscount
    community.py   # LouvainAgent (community partition + per-community greedy)
    dqn.py         # DQNAgent (Q-learning + replay + target net + ε-greedy)
    sarsa.py       # SARSAAgent (on-policy TD(0), linear function approximation)
    bandit.py      # LinUCBAgent (linear contextual bandit, UCB)
    s2v_dqn.py     # S2VDQNAgent (Structure2Vec + DQN, Khalil et al. '17)
docs/
  RL_AGENTS.md              # Algorithmic reference for DQN / SARSA / LinUCB
  RESULTS_ANALYSIS.md       # Per-episode learning curves for the original RL agents
  COMPREHENSIVE_BENCHMARK.md # Head-to-head comparison of all 14 agents
scripts/
  compare_all.py   # Unified benchmark runner — runs every agent, writes JSON + report
train_gnn.py       # GNN embedding training
train_rl.py        # RL episode runner (CLI)
```

## Quickstart

```bash
# 1. Train GNN embeddings (skip if node_embeddings.pt already exists)
python train_gnn.py

# 2. Run greedy MC agent (uses Apple MPS automatically on Mac)
python train_rl.py \
  --agent greedy_mc \
  --budget 10 \
  --mc-sims 100 \
  --agent-mc-rollouts 50 \
  --seed 42

# 3. Baselines and centrality heuristics
python train_rl.py --agent degree           --budget 10
python train_rl.py --agent random           --budget 10
python train_rl.py --agent pagerank         --budget 10
python train_rl.py --agent kshell           --budget 10
python train_rl.py --agent degree_discount  --budget 10

# 4. Lazy greedy / RIS / Community
python train_rl.py --agent celfpp  --budget 10 --agent-mc-rollouts 30
python train_rl.py --agent imm     --budget 10 --imm-theta-max 50000
python train_rl.py --agent louvain --budget 10 --agent-mc-rollouts 30

# 5. Learning agents (see docs/RL_AGENTS.md)
python train_rl.py --agent dqn     --budget 10 --episodes 50 --epsilon-decay 200
python train_rl.py --agent sarsa   --budget 10 --episodes 50 --epsilon-decay 200
python train_rl.py --agent linucb  --budget 10 --episodes 5  --linucb-alpha 1.0
python train_rl.py --agent s2v_dqn --budget 10 --episodes 30 --batch-size 16

# 6. Run the full head-to-head benchmark (writes docs/COMPREHENSIVE_BENCHMARK.md)
python scripts/compare_all.py --budget 10 --mc-sims 100 --device cpu
```

Results are saved to `results/<timestamp>_<agent>.json`. The unified benchmark runner additionally writes `results/benchmark_<timestamp>.json` and the markdown report.

## Device support

`--device auto` (default) selects **MPS** (Apple GPU) → CUDA → CPU automatically. Explicitly override with `--device mps|cuda|cpu`.

## Adding a new agent

1. Implement `BaseAgent` in `influence_max/agents/<algo>.py`
2. Register the strategy string in `build_agent()` in `monte_carlo.py`
3. Add any new fields to `AgentConfig` in `config.py`
4. Add the strategy to `--agent` CLI choices in `train_rl.py`
5. The episode loop in `train_rl.py` requires no changes
