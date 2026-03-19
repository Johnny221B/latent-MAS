#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/multi_train.py "$@"
