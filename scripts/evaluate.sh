#!/bin/bash
set -euo pipefail

CKPT_FOLDER="${CKPT_FOLDER:-outputs/gsm8k_qwen3-8b_20260319_102827}"

uv run --python .venv/bin/python src/cli/evaluate.py \
  --config "${CKPT_FOLDER}/config.yaml" \
  --checkpoint "${CKPT_FOLDER}/final_model.pt" \
  --max_samples -1 \
  --inference-mode chat_with_prefix \
  "$@"
