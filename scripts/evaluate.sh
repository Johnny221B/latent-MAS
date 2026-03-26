#!/bin/bash
set -euo pipefail

CKPT_FOLDER="${CKPT_FOLDER:-outputs/gsm8k_qwen3-4b_20260325_195232}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"

uv run --python .venv/bin/python torchrun \
  --master_port="${MASTER_PORT:-29612}" \
  --nproc_per_node=1 \
  src/cli/evaluate.py \
    --config "${CKPT_FOLDER}/config.yaml" \
    --checkpoint "${CKPT_FOLDER}/final_model.pt" \
    --eval-config configs/eval/${DATASET_NAME}.yaml \
    --output-dir "${CKPT_FOLDER}/${DATASET_NAME}" \
    "$@"
