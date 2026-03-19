#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/run_baseline_single_model.py \
  --model-name Qwen/Qwen3-8B \
  --max-samples -1 \
  --max-new-tokens 2048 \
  --batch-size 32 \
  --output-dir outputs/baselines \
  "$@"
