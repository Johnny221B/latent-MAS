#!/bin/bash
set -euo pipefail
# export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}

uv run --python .venv/bin/python torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  src/cli/train.py \
  --config configs/experiments/gsm8k_4agent.yaml \
  --max_samples 100000 \
  "$@"

# /home/chengzhi.ucsb/code/toby/latent-MAS/configs/experiments/gsm8k_5agent.yaml
# /home/chengzhi.ucsb/code/toby/latent-MAS/configs/experiments/gsm8k_4agent.yaml