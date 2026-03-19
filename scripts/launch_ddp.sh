#!/bin/bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun \
  --master_port=29500 \
  --nproc_per_node=2 \
  src/cli/multi_train.py \
  --config configs/experiments/gsm8k_3agent.yaml \
  "$@"
