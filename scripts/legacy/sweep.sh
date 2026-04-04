#!/bin/bash
set -euo pipefail

# Hyperparameter sweep for GSM8K 4-agent (Qwen3-4B)
# Usage:
#   bash scripts/sweep.sh                    # run all 8 experiments sequentially
#   bash scripts/sweep.sh r1                 # run only Round 1 (epoch sweep)
#   bash scripts/sweep.sh <config.yaml>      # run a single config

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SWEEP_DIR="configs/experiments/sweep"
NPROC="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29688}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

run_one() {
    local config="$1"
    local name
    name=$(basename "$config" .yaml)
    echo ""
    echo "=========================================="
    echo "  Running: $name"
    echo "  Config:  $config"
    echo "  GPUs:    $CUDA_VISIBLE_DEVICES"
    echo "  Time:    $(date)"
    echo "=========================================="
    echo ""
    uv run --python .venv/bin/python torchrun \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$NPROC" \
        src/cli/train.py \
        --config "$config"
}

if [ $# -eq 0 ]; then
    # Run all
    for config in "$SWEEP_DIR"/r*.yaml; do
        run_one "$config"
    done
elif [ -f "$1" ]; then
    # Single config file
    run_one "$1"
else
    # Filter by round prefix (r1, r2, r3, r4)
    round="$1"
    for config in "$SWEEP_DIR"/${round}_*.yaml; do
        run_one "$config"
    done
fi

echo ""
echo "Sweep complete!"
