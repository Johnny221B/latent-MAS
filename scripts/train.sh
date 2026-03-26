#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1

uv run --python .venv/bin/python torchrun \
  --master_port="${MASTER_PORT:-29633}" \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  src/cli/train.py \
  --config configs/experiments/am_deepseek_r1_distilled_5agent.yaml \
  --max_samples 100000
  "$@"
