# Latent Multi-Agent Communication Framework

Multi-agent collaboration via latent-space communication with learnable graph topology.

## Setup

```bash
# Create local environment
uv venv .venv

# Install dependencies
uv pip install --python .venv/bin/python -r requirements.txt

# Download model weights (example: Qwen3-0.5B)
uv run --python .venv/bin/python python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen3-0.5B'
AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./weights')
AutoTokenizer.from_pretrained(model_name, cache_dir='./weights')
"
```

## Quick Start

```bash
bash scripts/train.sh --config configs/experiments/gsm8k_3agent.yaml
```

## Project Structure

- `configs/` — Role definitions, graph priors, experiment configs
- `src/cli/` — Python entrypoints for training, evaluation, and baselines
- `src/models/` — Base model wrapper, compressor, agent
- `src/graph/` — Learnable adjacency matrix, DAG executor
- `src/communication/` — Latent message passing
- `src/losses/` — Task loss + graph regularization
- `src/pipeline/` — Top-level MultiAgentSystem
- `scripts/` — Shell wrappers only
- `.venv/` — Local virtual environment
- `weights/` — Model weights cache (gitignored)
