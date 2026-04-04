#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/run_baseline_paper_latentmas.py \
  --model-name Qwen/Qwen3-8B \
  --max-samples -1 \
  --latent-steps 10 \
  --max-new-tokens 2048 \
  --prompt sequential \
  --output-dir outputs/baselines \
  "$@"
