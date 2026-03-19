#!/bin/bash
set -euo pipefail

CKPT_FOLDER="${CKPT_FOLDER:-outputs/gsm8k_qwen3-8b_20260319_103852}"
WORKER="${WORKER:-2}"
MASTER_PORT="${MASTER_PORT:-29612}"

uv run --python .venv/bin/python torchrun \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${WORKER}" \
  src/cli/evaluate.py \
    --config "${CKPT_FOLDER}/config.yaml" \
    --checkpoint "${CKPT_FOLDER}/final_model.pt" \
    --max-new-tokens 16384 \
    --max_samples 100 \
    --inference-mode chat_with_prefix \
    --worker "${WORKER}" \
    "$@"
