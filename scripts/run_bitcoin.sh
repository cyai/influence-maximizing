#!/usr/bin/env bash
# =============================================================================
# Bitcoin Alpha generalisation experiment
# -----------------------------------------------------------------------------
# Runs the full IM pipeline on the soc-sign-bitcoin-alpha dataset and compares
# all agent families under identical conditions to the p2p-Gnutella08 results.
#
# Signed-network handling:
#   • Only RATING > 0 (trust) edges are used for IC diffusion.
#   • The GNN uses 7 signed features: pos/neg in-degree, pos/neg out-degree,
#     balance ratio, degree ratio, and PageRank on the positive subgraph.
#
# Outputs (all under bitcoin/):
#   bitcoin/soc-sign-bitcoin-alpha.csv   — raw dataset
#   bitcoin/node_embeddings.pt / .npy    — learned embeddings
#   bitcoin/embeddings_meta.json         — training metadata
#   bitcoin/checkpoints/best_model.pt    — GNN checkpoint
#   bitcoin/results/                     — per-agent JSON results
#   bitcoin/logs/                        — per-stage logs
#   bitcoin/docs/BITCOIN_BENCHMARK.md   — auto-generated comparison report
#
# Usage: bash scripts/run_bitcoin.sh
# Requires: run.sh was already executed once (venv + deps already installed).
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #
VENV_DIR="venv"
BITCOIN_DIR="bitcoin"
DATA_FILE="$BITCOIN_DIR/soc-sign-bitcoin-alpha.csv"
EMB_PT="$BITCOIN_DIR/node_embeddings.pt"
EMB_NPY="$BITCOIN_DIR/node_embeddings.npy"
META_JSON="$BITCOIN_DIR/embeddings_meta.json"
CKPT_DIR="$BITCOIN_DIR/checkpoints"
RESULTS_DIR="$BITCOIN_DIR/results"
LOG_DIR="$BITCOIN_DIR/logs"
REPORT_PATH="$BITCOIN_DIR/docs/BITCOIN_BENCHMARK.md"

BUDGET="${BUDGET:-10}"
MC_SIMS="${MC_SIMS:-50}"
MC_ROLLOUTS="${MC_ROLLOUTS:-20}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
IMM_THETA=15000

mkdir -p "$BITCOIN_DIR/docs" "$RESULTS_DIR" "$LOG_DIR" "$CKPT_DIR"

LOG_FILE="$LOG_DIR/run_bitcoin.log"
: > "$LOG_FILE"

log()  { printf "[%s] %s\n" "$(date +'%H:%M:%S')" "$*" | tee -a "$LOG_FILE"; }
ok()   { printf "[%s] OK   %s\n" "$(date +'%H:%M:%S')" "$*" | tee -a "$LOG_FILE"; }
err()  { printf "[%s] ERR  %s\n" "$(date +'%H:%M:%S')" "$*" | tee -a "$LOG_FILE"; }

stage() {
    printf "\n============================================================\n" | tee -a "$LOG_FILE"
    printf "%s\n" "$*" | tee -a "$LOG_FILE"
    printf "============================================================\n" | tee -a "$LOG_FILE"
}

# Activate the shared venv created by run.sh
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    err "Virtual environment not found at $VENV_DIR. Run 'bash run.sh' first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

OVERALL_START=$(date +%s)

# ---------------------------------------------------------------------------- #
# STAGE 1 — Download dataset
# ---------------------------------------------------------------------------- #
stage "STAGE 1 / 4: Download soc-sign-bitcoin-alpha"

# Note: SNAP filename is soc-sign-bitcoinalpha (no hyphen between bitcoin and alpha)
SNAP_URL="https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"

if [ -f "$DATA_FILE" ]; then
    log "Dataset already present at $DATA_FILE — skipping download."
else
    log "Downloading from $SNAP_URL ..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL "$SNAP_URL" -o "$DATA_FILE.gz"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$SNAP_URL" -O "$DATA_FILE.gz"
    else
        err "Neither curl nor wget found. Install one and retry."
        exit 1
    fi
    gunzip "$DATA_FILE.gz"
    ok "Downloaded: $DATA_FILE ($(wc -l < "$DATA_FILE") lines)"
fi

# Quick dataset stats
python - "$DATA_FILE" <<'PY' | tee -a "$LOG_FILE"
import csv, sys, collections
pos, neg = 0, 0
nodes = set()
with open(sys.argv[1], newline="") as f:
    for row in csv.reader(f):
        if not row or not row[0].strip().lstrip("-").isdigit(): continue
        if len(row) < 3: continue
        src, tgt, rating = int(row[0]), int(row[1]), int(row[2])
        nodes.update((src, tgt))
        if rating > 0: pos += 1
        else: neg += 1
print(f"  Nodes: {len(nodes):,}  |  Positive edges (trust): {pos:,}  |  Negative edges (distrust): {neg:,}")
print(f"  Positive edge fraction: {pos/(pos+neg)*100:.1f}%  — these carry influence in IC model")
PY

# ---------------------------------------------------------------------------- #
# STAGE 2 — Train GNN with signed features
# ---------------------------------------------------------------------------- #
stage "STAGE 2 / 4: Train GNN on Bitcoin Alpha (signed features)"

if [ -f "$EMB_PT" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    log "$EMB_PT already exists — skipping GNN training."
    log "    set FORCE_RETRAIN=1 to retrain from scratch."
else
    log "Training GraphSAGE+GAE with 7 signed-network features ..."
    log "    (pos/neg in-deg, pos/neg out-deg, balance, deg-ratio, PageRank)"
    GNN_EPOCHS="${GNN_EPOCHS:-1500}" python train_gnn.py \
        --data-path     "$DATA_FILE"  \
        --signed                      \
        --out-pt        "$EMB_PT"     \
        --out-npy       "$EMB_NPY"    \
        --out-meta      "$META_JSON"  \
        --checkpoint-dir "$CKPT_DIR" \
        2>&1 | tee "$LOG_DIR/train_gnn.log"
fi

if [ ! -f "$EMB_PT" ]; then
    err "$EMB_PT not produced — cannot continue."
    exit 1
fi
ok "Embeddings ready: $(ls -lh "$EMB_PT" | awk '{print $5}')"

# Print AUC from metadata
python - "$META_JSON" <<'PY' | tee -a "$LOG_FILE"
import json, sys
m = json.loads(open(sys.argv[1]).read())
print(f"  Best val AUC : {m['best_val_auc']:.4f}  (epoch {m['best_epoch']} / {m['epochs_trained']})")
print(f"  Nodes        : {m['num_nodes']:,}   Features per node: {m['num_features']}")
PY

# ---------------------------------------------------------------------------- #
# STAGE 3 — Per-agent runs
# ---------------------------------------------------------------------------- #
stage "STAGE 3 / 4: Individual agent runs on Bitcoin Alpha"

COMMON="--data-path $DATA_FILE --embeddings $EMB_PT --device $DEVICE --seed $SEED --results-dir $RESULTS_DIR"

run_agent() {
    local name="$1"; shift
    local agent_log="$LOG_DIR/agent_${name}.log"
    log "  → ${name}"
    local t0; t0=$(date +%s)
    if python train_rl.py $COMMON "$@" > "$agent_log" 2>&1; then
        local dt=$(( $(date +%s) - t0 ))
        # Extract final spread from JSON result
        local spread
        spread=$(python -c "
import json, glob, os
files = sorted(glob.glob('${RESULTS_DIR}/*_${name}.json'))
if files:
    d = json.loads(open(files[-1]).read())
    print(f\"{d['summary']['mean_spread']:.2f}\")
else:
    print('?')
" 2>/dev/null || echo "?")
        ok "  ← ${name} done in ${dt}s  spread=${spread}"
    else
        local dt=$(( $(date +%s) - t0 ))
        err "  ← ${name} FAILED after ${dt}s — see ${agent_log}"
    fi
}

# Baselines & centrality
run_agent random          --agent random          --budget "$BUDGET" --mc-sims "$MC_SIMS"
run_agent degree          --agent degree          --budget "$BUDGET" --mc-sims "$MC_SIMS"
run_agent pagerank        --agent pagerank        --budget "$BUDGET" --mc-sims "$MC_SIMS"
run_agent kshell          --agent kshell          --budget "$BUDGET" --mc-sims "$MC_SIMS"
run_agent degree_discount --agent degree_discount --budget "$BUDGET" --mc-sims "$MC_SIMS"

# RIS
run_agent imm --agent imm --budget "$BUDGET" --mc-sims "$MC_SIMS" --imm-theta-max "$IMM_THETA"

# Simulation-based greedy
run_agent celf      --agent celf      --budget "$BUDGET" --mc-sims "$MC_SIMS" --agent-mc-rollouts "$MC_ROLLOUTS"
run_agent greedy_mc --agent greedy_mc --budget "$BUDGET" --mc-sims "$MC_SIMS" --agent-mc-rollouts "$MC_ROLLOUTS"

# Learning agents
run_agent linucb  --agent linucb  --budget "$BUDGET" --mc-sims "$MC_SIMS" --episodes 5
run_agent sarsa   --agent sarsa   --budget "$BUDGET" --mc-sims "$MC_SIMS" --episodes 10 --epsilon-decay 100
run_agent dqn     --agent dqn     --budget "$BUDGET" --mc-sims "$MC_SIMS" --episodes 10 --epsilon-decay 100 --batch-size 32
run_agent s2v_dqn --agent s2v_dqn --budget "$BUDGET" --mc-sims "$MC_SIMS" --episodes 10 --batch-size 8

# ---------------------------------------------------------------------------- #
# STAGE 4 — Comprehensive benchmark
# ---------------------------------------------------------------------------- #
stage "STAGE 4 / 4: Comprehensive benchmark (compare_all.py)"

log "Writing report to $REPORT_PATH"

if python scripts/compare_all.py \
        --data-path "$DATA_FILE" \
        --embeddings "$EMB_PT" \
        --device "$DEVICE" --seed "$SEED" \
        --budget "$BUDGET" \
        --mc-sims "$MC_SIMS" \
        --agent-mc-rollouts "$MC_ROLLOUTS" \
        --episodes 1 \
        --epsilon-decay 100 \
        --imm-theta-max "$IMM_THETA" \
        --batch-size 32 \
        --results-dir "$RESULTS_DIR" \
        --report-path "$REPORT_PATH" \
        --exclude celfpp louvain betweenness \
        > "$LOG_DIR/compare_all.log" 2>&1; then
    ok "Benchmark complete — see $REPORT_PATH"
else
    err "Benchmark FAILED — see $LOG_DIR/compare_all.log"
fi

# ---------------------------------------------------------------------------- #
# Summary
# ---------------------------------------------------------------------------- #
OVERALL_END=$(date +%s)
ELAPSED=$(( OVERALL_END - OVERALL_START ))

stage "Bitcoin Alpha experiment complete in $(( ELAPSED/60 ))m $(( ELAPSED%60 ))s"

LATEST_BENCHMARK="$(ls -t "$RESULTS_DIR"/benchmark_*.json 2>/dev/null | head -n1 || true)"

cat <<EOF | tee -a "$LOG_FILE"

Dataset : soc-sign-bitcoin-alpha  (positive-trust edges only for IC)
Budget  : k = $BUDGET   MC sims = $MC_SIMS   seed = $SEED

Artifacts:
  $EMB_PT          — node embeddings (signed features)
  $META_JSON       — GNN training metadata
  $RESULTS_DIR/    — per-agent JSON results
  ${LATEST_BENCHMARK:-(none)}   — benchmark JSON
  $REPORT_PATH     — markdown comparison report
  $LOG_DIR/        — per-stage logs

Top results:
EOF

if [ -n "${LATEST_BENCHMARK:-}" ] && [ -f "$LATEST_BENCHMARK" ]; then
    python - "$LATEST_BENCHMARK" <<'PY' | tee -a "$LOG_FILE"
import json, sys
data = json.loads(open(sys.argv[1]).read())
results = [r for r in data["results"] if "error" not in r]
results.sort(key=lambda r: -r["mean_spread"])
print(f"  {'Family':<14} {'Algorithm':<18} {'Spread':>10} {'Time(s)':>10}")
print("  " + "-" * 56)
for r in results:
    print(f"  {r['family']:<14} {r['label']:<18} {r['mean_spread']:>10.2f} {r['elapsed_sec']:>10.1f}")
PY
fi

ok "Done."
