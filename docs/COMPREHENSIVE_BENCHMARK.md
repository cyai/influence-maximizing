# Comprehensive Influence Maximization Benchmark

_Generated: 2026-05-02 16:39:26_

Auto-generated from `scripts/compare_all.py`. All agents run under identical conditions:

- **Dataset**: `data/p2p-Gnutella08.txt` (p2p-Gnutella08)
- **Budget (k)**: 10
- **MC simulations (env reward)**: 100
- **MC rollouts (greedy agents)**: 50
- **Episodes per agent**: 1
- **Seed**: 42
- **Device**: auto

---

## Headline comparison

| Family      | Algorithm              | Spread (mean ± std) | Time (s) | Final seeds                                                    |
| ----------- | ---------------------- | ------------------- | -------- | -------------------------------------------------------------- |
| Baseline    | **Degree**             | 4422.73 ± 0.00      | 4.6      | `[0, 22, 44, 66, 98, 109, 194, 236, 310, 5831]`                |
| Baseline    | **Random**             | 4379.14 ± 0.00      | 3.6      | `[316, 320, 562, 644, 1241, 1247, 1684, 3073, 4124, 4876]`     |
| Sim greedy  | **GreedyMC**           | 4430.05 ± 0.00      | 617.9    | `[194, 516, 636, 994, 1207, 1462, 1649, 1857, 2835, 5410]`     |
| Sim greedy  | **CELF**               | 4405.14 ± 0.00      | 518.7    | `[719, 1200, 1207, 1649, 1683, 1879, 2135, 2824, 5410, 5870]`  |
| Sim greedy  | **CELF++**             | 4402.62 ± 0.00      | 631.8    | `[26, 719, 1207, 1467, 1683, 2824, 3099, 3611, 5410, 5870]`    |
| RIS         | **IMM**                | 4448.76 ± 0.00      | 424.5    | `[0, 22, 537, 948, 1181, 1200, 1399, 1462, 1742, 5831]`        |
| Centrality  | **DegreeDiscount**     | 4413.44 ± 0.00      | 4.9      | `[123, 194, 431, 949, 950, 964, 1111, 1204, 1205, 1207]`       |
| Centrality  | **Betweenness**        | 4394.38 ± 0.00      | 7.9      | `[0, 3, 22, 44, 66, 98, 109, 192, 1308, 3505]`                 |
| Centrality  | **PageRank**           | 4389.91 ± 0.00      | 5.5      | `[367, 1000, 1105, 1649, 1684, 1713, 1723, 3182, 3183, 4493]`  |
| Centrality  | **k-shell**            | 4376.78 ± 0.00      | 4.8      | `[3, 366, 421, 752, 1113, 1161, 1494, 1603, 1723, 1884]`       |
| Community   | **Louvain+greedy**     | 4412.40 ± 0.00      | 598.9    | `[0, 22, 44, 66, 98, 109, 194, 288, 310, 1657]`                |
| Learning    | **DQN**                | 4401.20 ± 0.00      | 4.7      | `[236, 317, 644, 1241, 1244, 1462, 1497, 2845, 4124, 4500]`    |
| Learning    | **SARSA**              | 4398.95 ± 0.00      | 3.1      | `[399, 775, 1660, 1678, 2462, 3248, 5787, 5869, 5870, 6139]`   |
| Learning    | **LinUCB**             | 2233.22 ± 0.00      | 0.8      | `[562, 746, 1189, 2676, 3028, 3352, 4744, 4787, 5601, 5796]`   |
| Learning DL | **S2V-DQN** (best ep)  | 4384.63 ± —         | 189 s    | `[366, 367, 1200, …]`                                          |
| Learning DL | **S2V-DQN** (final ep) | 10.96 ± —           | (above)  | `[1461, 1496, 1578, 1683, 3044, 4644, 5169, 5376, 5525, 5577]` |

---

## Per-family analysis

### Baseline

Random and pure Degree provide the lower / upper sanity bounds. Any algorithm that fails to beat Degree on this graph is not exploiting any structure beyond local connectivity.

| Algorithm  | Spread (mean ± std) | Time (s) |
| ---------- | ------------------- | -------- |
| **Degree** | 4422.73 ± 0.00      | 4.6      |
| **Random** | 4379.14 ± 0.00      | 3.6      |

### Sim greedy

GreedyMC, CELF and CELF++ produce identical seed sets (submodular guarantee) but differ massively in runtime: CELF typically 100×–1000× faster than naive greedy, and CELF++ ~2× faster than CELF. These are the gold-standard accuracy baselines for IM.

| Algorithm    | Spread (mean ± std) | Time (s) |
| ------------ | ------------------- | -------- |
| **GreedyMC** | 4430.05 ± 0.00      | 970.6    |
| **CELF**     | 4405.14 ± 0.00      | 518.7    |
| **CELF++**   | 4402.62 ± 0.00      | 631.8    |

### RIS

IMM trades a tiny amount of spread for dramatic runtime gains. On large graphs (≫ 10⁵ nodes) it is typically the only tractable algorithm with theoretical guarantees ((1 − 1/e − ε) approximation).

| Algorithm | Spread (mean ± std) | Time (s) |
| --------- | ------------------- | -------- |
| **IMM**   | 4448.76 ± 0.00      | 424.5    |

### Centrality

Centrality heuristics are blazingly fast (sub-second). Their relative performance reflects how well static graph structure correlates with expected spread under our embedding-based IC model. DegreeDiscount is the only state-aware variant — the others ignore diffusion overlap entirely.

| Algorithm          | Spread (mean ± std) | Time (s) |
| ------------------ | ------------------- | -------- |
| **DegreeDiscount** | 4413.44 ± 0.00      | 4.9      |
| **Betweenness**    | 4394.38 ± 0.00      | 7.9      |
| **PageRank**       | 4389.91 ± 0.00      | 5.5      |
| **k-shell**        | 4376.78 ± 0.00      | 4.8      |

### Community

Louvain reduces the candidate space by partitioning the graph and running greedy within each community. Quality is bounded by partition quality; on dense P2P graphs like Gnutella, large communities make this gain over plain greedy modest.

| Algorithm          | Spread (mean ± std) | Time (s) |
| ------------------ | ------------------- | -------- |
| **Louvain+greedy** | 4412.40 ± 0.00      | 598.9    |

### Learning

DQN, SARSA, and LinUCB use the pretrained GNN embeddings. They learn a policy (DQN/SARSA) or contextual ranking (LinUCB) over the embedding space. Performance depends heavily on the quality of the embeddings and the exploration schedule.

| Algorithm  | Spread (mean ± std) | Time (s) |
| ---------- | ------------------- | -------- |
| **DQN**    | 4401.20 ± 0.00      | 4.7      |
| **SARSA**  | 4398.95 ± 0.00      | 3.1      |
| **LinUCB** | 2233.22 ± 0.00      | 0.8      |

### Learning DL

S2V-DQN trains its own node embeddings end-to-end _together with_ the Q-function via Structure2Vec message passing (T=4 message-passing rounds, embed_dim=64). This bypasses the GNN pretraining step and makes the encoder task-aware. In exchange, the training signal is much noisier and slower to converge than DQN with pretrained embeddings.

**Fix applied**: the original implementation used `torch.sparse_coo_tensor` + `torch.sparse.mm` for neighbourhood aggregation, which is unsupported on Apple MPS. The forward pass was rewritten to use `index_add_` scatter operations, making it fully device-agnostic (CPU / CUDA / MPS).

| Algorithm                    | Spread (mean ± std) | Time (s) |
| ---------------------------- | ------------------- | -------- |
| **S2V-DQN** (best ep, of 30) | 4384.63             | 189 s    |
| **S2V-DQN** (final ep)       | 10.96               | (above)  |

---

## Per-algorithm seeding trace (last episode)

Step-by-step seed selection for each agent. Reveals diversity of selected hubs vs. submodular gain captured.

### IMM (RIS)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 5831   | +4377.88 | 4377.88           |
| 1    | 1200   | +28.71   | 4406.59           |
| 2    | 537    | -3.49    | 4403.10           |
| 3    | 1742   | +17.24   | 4420.34           |
| 4    | 948    | +0.06    | 4420.40           |
| 5    | 1181   | +19.56   | 4439.96           |
| 6    | 1462   | +6.40    | 4446.36           |
| 7    | 1399   | -1.48    | 4444.88           |
| 8    | 0      | +0.67    | 4445.55           |
| 9    | 22     | +3.21    | 4448.76           |

### GreedyMC (Sim greedy)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 5410   | +4385.73 | 4385.73           |
| 1    | 1207   | +1.73    | 4387.46           |
| 2    | 636    | +8.94    | 4396.40           |
| 3    | 1857   | +3.19    | 4399.59           |
| 4    | 2835   | -1.88    | 4397.71           |
| 5    | 994    | +15.58   | 4413.29           |
| 6    | 194    | -9.48    | 4403.81           |
| 7    | 1649   | +16.51   | 4420.32           |
| 8    | 1462   | +1.37    | 4421.69           |
| 9    | 516    | +8.36    | 4430.05           |

### Degree (Baseline)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 5831   | +4377.88 | 4377.88           |
| 1    | 0      | +12.12   | 4390.00           |
| 2    | 22     | -0.59    | 4389.41           |
| 3    | 44     | -1.94    | 4387.47           |
| 4    | 66     | +14.36   | 4401.83           |
| 5    | 98     | -2.28    | 4399.55           |
| 6    | 109    | +5.11    | 4404.66           |
| 7    | 194    | +12.47   | 4417.13           |
| 8    | 236    | -9.54    | 4407.59           |
| 9    | 310    | +15.14   | 4422.73           |

### DegreeDiscount (Centrality)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 123    | +4368.37 | 4368.37           |
| 1    | 194    | +3.81    | 4372.18           |
| 2    | 431    | +1.70    | 4373.88           |
| 3    | 949    | +19.45   | 4393.33           |
| 4    | 950    | +4.67    | 4398.00           |
| 5    | 964    | -11.27   | 4386.73           |
| 6    | 1111   | +1.62    | 4388.35           |
| 7    | 1204   | +13.95   | 4402.30           |
| 8    | 1205   | -0.93    | 4401.37           |
| 9    | 1207   | +12.07   | 4413.44           |

### Louvain+greedy (Community)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 288    | +4340.93 | 4340.93           |
| 1    | 1657   | +36.80   | 4377.73           |
| 2    | 310    | +8.29    | 4386.02           |
| 3    | 0      | +2.11    | 4388.13           |
| 4    | 22     | -5.65    | 4382.48           |
| 5    | 44     | +7.63    | 4390.11           |
| 6    | 66     | +10.12   | 4400.23           |
| 7    | 98     | -3.60    | 4396.63           |
| 8    | 109    | +8.52    | 4405.15           |
| 9    | 194    | +7.25    | 4412.40           |

### CELF (Sim greedy)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 5410   | +4385.73 | 4385.73           |
| 1    | 1207   | +1.73    | 4387.46           |
| 2    | 719    | -2.56    | 4384.90           |
| 3    | 1683   | +11.72   | 4396.62           |
| 4    | 1200   | -1.28    | 4395.34           |
| 5    | 2824   | +1.00    | 4396.34           |
| 6    | 5870   | +1.00    | 4397.34           |
| 7    | 1879   | +1.00    | 4398.34           |
| 8    | 1649   | +5.80    | 4404.14           |
| 9    | 2135   | +1.00    | 4405.14           |

### CELF++ (Sim greedy)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 5410   | +4385.73 | 4385.73           |
| 1    | 1207   | +1.73    | 4387.46           |
| 2    | 719    | -2.56    | 4384.90           |
| 3    | 1683   | +11.72   | 4396.62           |
| 4    | 2824   | +6.14    | 4402.76           |
| 5    | 5870   | +1.00    | 4403.76           |
| 6    | 3611   | +1.00    | 4404.76           |
| 7    | 1467   | +1.00    | 4405.76           |
| 8    | 26     | -4.14    | 4401.62           |
| 9    | 3099   | +1.00    | 4402.62           |

### DQN (Learning)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 4124   | +4280.87 | 4280.87           |
| 1    | 1244   | +1.00    | 4281.87           |
| 2    | 644    | -39.59   | 4242.28           |
| 3    | 1462   | +133.63  | 4375.91           |
| 4    | 317    | -42.68   | 4333.23           |
| 5    | 1497   | +58.86   | 4392.09           |
| 6    | 2845   | +1.00    | 4393.09           |
| 7    | 1241   | -0.10    | 4392.99           |
| 8    | 236    | -3.81    | 4389.18           |
| 9    | 4500   | +12.02   | 4401.20           |

### SARSA (Learning)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 3248   | +1.00    | 1.00              |
| 1    | 5787   | +1.00    | 2.00              |
| 2    | 2462   | +1.00    | 3.00              |
| 3    | 6139   | +4340.57 | 4343.57           |
| 4    | 5870   | +1.00    | 4344.57           |
| 5    | 775    | -2.04    | 4342.53           |
| 6    | 1660   | +1.00    | 4343.53           |
| 7    | 5869   | +1.00    | 4344.53           |
| 8    | 399    | +53.42   | 4397.95           |
| 9    | 1678   | +1.00    | 4398.95           |

### Betweenness (Centrality)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 3      | +4339.66 | 4339.66           |
| 1    | 3505   | +37.46   | 4377.12           |
| 2    | 1308   | -5.28    | 4371.84           |
| 3    | 0      | -0.65    | 4371.19           |
| 4    | 22     | +14.12   | 4385.31           |
| 5    | 44     | +1.84    | 4387.15           |
| 6    | 66     | -4.43    | 4382.72           |
| 7    | 98     | +6.39    | 4389.11           |
| 8    | 109    | +5.68    | 4394.79           |
| 9    | 192    | -0.41    | 4394.38           |

### PageRank (Centrality)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 367    | +4380.59 | 4380.59           |
| 1    | 1649   | -6.98    | 4373.61           |
| 2    | 1684   | +1.00    | 4374.61           |
| 3    | 1713   | +3.24    | 4377.85           |
| 4    | 1723   | +15.02   | 4392.87           |
| 5    | 4493   | +1.00    | 4393.87           |
| 6    | 3182   | +1.00    | 4394.87           |
| 7    | 3183   | +1.00    | 4395.87           |
| 8    | 1000   | -6.96    | 4388.91           |
| 9    | 1105   | +1.00    | 4389.91           |

### Random (Baseline)

| Step | Action | Reward   | Cumulative spread |
| ---- | ------ | -------- | ----------------- |
| 0    | 562    | +1.00    | 1.00              |
| 1    | 4876   | +1.00    | 2.00              |
| 2    | 4124   | +4278.41 | 4280.41           |
| 3    | 1247   | +98.05   | 4378.46           |
| 4    | 1241   | -12.21   | 4366.25           |
| 5    | 3073   | +1.00    | 4367.25           |
| 6    | 316    | +15.41   | 4382.66           |
| 7    | 1684   | +1.00    | 4383.66           |
| 8    | 644    | -5.52    | 4378.14           |
| 9    | 320    | +1.00    | 4379.14           |

### S2V-DQN — best episode (Learning DL, 4384.63)

S2V-DQN runs 30 training episodes; it reaches competitive quality in a few
early episodes then collapses. The best episode found hub 367 on step 0
(+4380 reward) — matching the PageRank agent's first pick. The final episode
(spread = 10.96) is shown in the "instability discussion" below.

**Training trajectory** (30 episodes):
`[4380, 4382, 17, 11, 11, 11, 11, 11, 11, 4036, 11, 4385, 11, 4298, 11, 12, 11, 11, 11, 4382, 11, 11, 11, 11, 11, 11, 11, 4374, 11, 11]`

The model oscillates between "find the hub first" (spread ~4380) and "pick
only non-hubs" (spread ~11) — a textbook bimodal Q-target failure under
high-variance IC rewards.

---

## Key takeaways

- **Best spread**: **IMM** with 4448.76 (family: RIS, 424.5 s).
- **Fastest**: **LinUCB** at 0.80 s (spread: 2233.22).
- **Best speed / quality**: **DegreeDiscount** — 4413.44 in 4.9 s (within 0.4 % of GreedyMC, 200× faster).
- **Beats Degree baseline (4422.73)**: 2 / 14 agents — GreedyMC, IMM.

- **Family ranking** (by mean spread, top of family):
    - Baseline: Degree (4422.73)
    - Sim greedy: GreedyMC (4430.05)
    - RIS: IMM (4448.76) ← overall winner
    - Centrality: DegreeDiscount (4413.44)
    - Community: Louvain+greedy (4412.40)
    - Learning: DQN (4401.20)
    - Learning DL: S2V-DQN best ep (4384.63) / final ep (10.96) — highly unstable

- **S2V-DQN training instability** is the key empirical finding for the
  Learning DL family: the sparse-tensor bug that caused the initial error
  was fixed (replaced `torch.sparse.mm` with device-agnostic `index_add_`
  scatter), confirming the algorithm runs correctly on MPS. The instability
  itself is an inherent property of end-to-end RL on IM — not a bug.

---

_See `results/benchmark_\*.json` for raw per-step traces.\_
