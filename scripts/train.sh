#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python torchrun \
  --master_port="${MASTER_PORT:-29611}" \
  --nproc_per_node="${NPROC_PER_NODE:-4}" \
  src/cli/train.py \
  --config configs/experiments/gsm8k_5agent.yaml \
  "$@"
