#!/bin/bash
set -euo pipefail

export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

/usr/bin/python -m torch.distributed.run \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  src/cli/train.py \
  --config configs/experiments/competition_math_4b_init_hidden.yaml \
  "$@"
