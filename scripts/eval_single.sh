#!/bin/bash
set -euo pipefail

uv run --python .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8080 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --api-key EMPTY \
  --tensor-parallel-size 4 

uv run --python .venv/bin/python src/cli/run_baseline_single_model.py \
    --model-name Qwen/Qwen3-8B \
    --max-samples 100 \
    --max-new-tokens 15500 \
    --worker-count 16 \
    --base-url http://127.0.0.1:8000 \
    --api-key EMPTY \
    --reuse-server \
    --output-dir outputs/baselines
  "$@"
