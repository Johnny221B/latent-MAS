#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/plot_loss.py \
  --input outputs/gsm8k_qwen3-8b_20260319_052516/loss_log.csv \
  --output outputs/epoch_loss_curve.png \
  "$@"
