#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
LOG_DIR="logs"
RESULTS_DIR="results"

BUDGET="${BUDGET:-10}"
MC_SIMS="${MC_SIMS:-50}"
MC_ROLLOUTS="${MC_ROLLOUTS:-20}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"

# you can change this value to the max number of RR-sets for IMM
IMM_THETA="${IMM_THETA_MAX:-50000}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
LOG_FILE="$LOG_DIR/run.log"
: > "$LOG_FILE"

# Colour output only when stdout is a real TTY
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD="\033[1m"; CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"
    RED="\033[31m"; RST="\033[0m"
else
    BOLD=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; RST=""
fi

log()  { printf "%b[%s]%b %s\n"      "$CYAN"   "$(date +'%H:%M:%S')" "$RST" "$*" | tee -a "$LOG_FILE"; }
ok()   { printf "%b[%s] OK%b   %s\n" "$GREEN"  "$(date +'%H:%M:%S')" "$RST" "$*" | tee -a "$LOG_FILE"; }
warn() { printf "%b[%s] WARN%b %s\n" "$YELLOW" "$(date +'%H:%M:%S')" "$RST" "$*" | tee -a "$LOG_FILE"; }
err()  { printf "%b[%s] ERR%b  %s\n" "$RED"    "$(date +'%H:%M:%S')" "$RST" "$*" | tee -a "$LOG_FILE"; }

stage() {
    printf "\n%b============================================================%b\n" "$BOLD" "$RST" | tee -a "$LOG_FILE"
    printf "%b%s%b\n" "$BOLD" "$*" "$RST" | tee -a "$LOG_FILE"
    printf "%b============================================================%b\n" "$BOLD" "$RST" | tee -a "$LOG_FILE"
}

OVERALL_START=$(date +%s)

stage "STAGE 1 / 6: System packages"

install_apt() {
    local SUDO=""
    if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update -y
        DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
            python3 python3-venv python3-pip python3-dev build-essential ca-certificates curl
    else
        warn "apt-get not found — assuming required system packages are pre-installed."
    fi
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 \
        || ! "$PYTHON_BIN" -c 'import venv' 2>/dev/null \
        || ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    log "Bootstrapping Python via apt-get ..."
    install_apt 2>&1 | tee -a "$LOG_DIR/apt.log"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    err "$PYTHON_BIN not available even after apt install. Aborting."
    exit 1
fi
ok "Python: $($PYTHON_BIN --version 2>&1)"

stage "STAGE 2 / 6: Python virtual environment"

if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    log "Creating venv in $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    log "Reusing existing venv at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade --quiet pip wheel setuptools 2>&1 | tee -a "$LOG_DIR/pip.log"
ok "venv ready ($(python -V 2>&1))"

stage "STAGE 3 / 6: Python dependencies"

log "Installing CPU PyTorch from https://download.pytorch.org/whl/cpu"
pip install --quiet --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1.0" 2>&1 | tee -a "$LOG_DIR/pip.log"

log "Installing project requirements (numpy, sklearn, tqdm, networkx, torch_geometric)"
pip install --quiet -r requirements.txt 2>&1 | tee -a "$LOG_DIR/pip.log"

log "Verifying imports"
python - <<'PY' | tee -a "$LOG_DIR/pip.log"
import torch, torch_geometric, numpy as np, sklearn, tqdm, networkx as nx
print(f"  torch           : {torch.__version__}")
print(f"  torch_geometric : {torch_geometric.__version__}")
print(f"  numpy           : {np.__version__}")
print(f"  sklearn         : {sklearn.__version__}")
print(f"  tqdm            : {tqdm.__version__}")
print(f"  networkx        : {nx.__version__}")
print(f"  device          : {'cuda' if torch.cuda.is_available() else 'cpu'}")
PY
ok "Dependencies installed"

stage "STAGE 4 / 6: GNN embedding training"

if [ -f "node_embeddings.pt" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    log "node_embeddings.pt already present — skipping GNN training."
    log "    set FORCE_RETRAIN=1 to retrain from scratch."
else
    log "Training GraphSAGE+GAE encoder (can take several minutes on CPU)..."
    GNN_EPOCHS="${GNN_EPOCHS:-1500}" python train_gnn.py 2>&1 | tee "$LOG_DIR/train_gnn.log"
fi

if [ ! -f "node_embeddings.pt" ]; then
    err "node_embeddings.pt not produced — cannot continue."
    exit 1
fi
ok "Embeddings ready: $(ls -lh node_embeddings.pt | awk '{print $5}')"

if [ "${SKIP_INDIVIDUAL:-0}" = "1" ]; then
    stage "STAGE 5 / 6: Individual agent runs  [SKIPPED via SKIP_INDIVIDUAL=1]"
else
    stage "STAGE 5 / 6: Individual agent runs"

    run_agent() {
        local name="$1"; shift
        local agent_log="$LOG_DIR/agent_${name}.log"
        log "  → ${name} :: train_rl.py $*"
        local t0; t0=$(date +%s)
        if python train_rl.py --device "$DEVICE" --seed "$SEED" "$@" > "$agent_log" 2>&1; then
            local dt=$(( $(date +%s) - t0 ))
            ok "  ← ${name} done in ${dt}s  (log: ${agent_log})"
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

    # Reverse Influence Sampling — IMM_THETA capped to avoid OOM in Docker
    run_agent imm --agent imm --budget "$BUDGET" --mc-sims "$MC_SIMS" --imm-theta-max "$IMM_THETA"

    # Simulation-based greedy
    run_agent celf      --agent celf      --budget "$BUDGET" --mc-sims "$MC_SIMS" --agent-mc-rollouts "$MC_ROLLOUTS"
    run_agent greedy_mc --agent greedy_mc --budget "$BUDGET" --mc-sims "$MC_SIMS" --agent-mc-rollouts "$MC_ROLLOUTS"

    # Learning agents
    run_agent linucb  --agent linucb  --budget "$BUDGET" --mc-sims "$MC_SIMS" --episodes 5
    run_agent sarsa   --agent sarsa   --budget "$BUDGET" --mc-sims "$MC_SIMS" --episodes 10 --epsilon-decay 100
    run_agent dqn     --agent dqn     --budget "$BUDGET" --mc-sims "$MC_SIMS" --episodes 10 --epsilon-decay 100 --batch-size 32

    # Learning DL
    run_agent s2v_dqn --agent s2v_dqn --budget "$BUDGET" --mc-sims "$MC_SIMS" --episodes 10 --batch-size 8
fi

if [ "${SKIP_BENCHMARK:-0}" = "1" ]; then
    stage "STAGE 6 / 6: Comprehensive benchmark  [SKIPPED via SKIP_BENCHMARK=1]"
else
    stage "STAGE 6 / 6: Comprehensive benchmark"

    log "Running scripts/compare_all.py (excluding celfpp, louvain, betweenness)"
    log "    Report  → docs/RUN_REPORT.md"
    log "    Raw JSON → results/benchmark_<timestamp>.json"

    if python scripts/compare_all.py \
            --device "$DEVICE" --seed "$SEED" \
            --budget "$BUDGET" \
            --mc-sims "$MC_SIMS" \
            --agent-mc-rollouts "$MC_ROLLOUTS" \
            --episodes 1 \
            --epsilon-decay 100 \
            --imm-theta-max "$IMM_THETA" \
            --batch-size 32 \
            --report-path docs/RUN_REPORT.md \
            --exclude celfpp louvain betweenness \
            > "$LOG_DIR/compare_all.log" 2>&1; then
        ok "Benchmark complete — see docs/RUN_REPORT.md"
    else
        err "Benchmark FAILED — see $LOG_DIR/compare_all.log"
    fi
fi

# ---------------------------------------------------------------------------- #
# Final summary
# ---------------------------------------------------------------------------- #
OVERALL_END=$(date +%s)
ELAPSED=$(( OVERALL_END - OVERALL_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

stage "Pipeline complete in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"

LATEST_BENCHMARK="$(ls -t $RESULTS_DIR/benchmark_*.json 2>/dev/null | head -n1 || true)"

cat <<EOF | tee -a "$LOG_FILE"
Artifacts produced:
  • node_embeddings.pt / .npy     — learned GNN node embeddings
  • embeddings_meta.json          — training metadata
  • checkpoints/best_model.pt     — best GNN checkpoint
  • ${RESULTS_DIR}/<ts>_<agent>.json  — per-agent result (one per agent)
  • ${LATEST_BENCHMARK:-(none)}       — unified benchmark JSON
  • docs/RUN_REPORT.md            — auto-generated comparison report
  • docs/COMPREHENSIVE_BENCHMARK.md  — curated reference (full 14-agent run)
  • ${LOG_DIR}/                   — per-stage logs

Top-6 algorithms by spread:
EOF

if [ -n "${LATEST_BENCHMARK:-}" ] && [ -f "$LATEST_BENCHMARK" ]; then
    python - "$LATEST_BENCHMARK" <<'PY' | tee -a "$LOG_FILE"
import json, sys
data = json.loads(open(sys.argv[1]).read())
results = [r for r in data["results"] if "error" not in r]
results.sort(key=lambda r: -r["mean_spread"])
print(f"  {'Family':<14} {'Algorithm':<18} {'Spread':>10} {'Time(s)':>10}")
print("  " + "-" * 56)
for r in results[:6]:
    print(f"  {r['family']:<14} {r['label']:<18} {r['mean_spread']:>10.2f} {r['elapsed_sec']:>10.1f}")
PY
fi

ok "All done."
