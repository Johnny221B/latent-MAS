#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/run_pipeline.py "$@"
