#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/run_pipeline.py \
  --hf-home .cache/huggingface \
  --cuda-visible-devices 0,1 \
  "$@"
