#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python torchrun \
  --master_port="${MASTER_PORT:-29611}" \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  src/cli/train.py \
  --config configs/experiments/competition_math_5agent.yaml \
  "$@"
