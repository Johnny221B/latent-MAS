#!/bin/bash
set -euo pipefail
uv run --python .venv/bin/python src/cli/plot_loss.py "$@"
