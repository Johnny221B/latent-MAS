#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/evaluate.py \
  --config outputs/gsm8k_qwen3-8b_20260319_052516/config.yaml \
  --checkpoint outputs/gsm8k_qwen3-8b_20260319_052516/final_model.pt \
  --max_samples -1 \
  --max-new-tokens 2048 \
  --inference-mode chat_with_prefix \
  "$@"
