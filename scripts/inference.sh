#!/bin/bash
set -euo pipefail

CKPT_FOLDER="${CKPT_FOLDER:-outputs/gsm8k_qwen3-8b_20260321_115253}"
WORKER="${WORKER:-1}"
MASTER_PORT="${MASTER_PORT:-29613}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_FOLDER}/inference_$(date +%Y%m%d_%H%M%S)}"

uv run --python .venv/bin/python torchrun \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${WORKER}" \
  src/cli/evaluate.py \
    --config "${CKPT_FOLDER}/config.yaml" \
    --checkpoint "${CKPT_FOLDER}/final_model.pt" \
    --max-new-tokens 16384 \
    --inference-mode chat_with_prefix \
    --worker "${WORKER}" \
    --output-dir "${OUTPUT_DIR}" \
    --question "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?" \
    "$@"