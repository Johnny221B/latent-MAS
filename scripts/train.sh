#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/train.py \
  --config configs/experiments/gsm8k_3agent.yaml \
  "$@"
