# Latent Multi-Agent Communication Framework

Multi-agent collaboration via latent-space communication with learnable graph topology.

## Setup

```bash
# Create local environment
python -m venv envs/venv
source envs/venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model weights (example: Qwen3-0.5B)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen3-0.5B'
AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./weights')
AutoTokenizer.from_pretrained(model_name, cache_dir='./weights')
"
```

## Quick Start

```bash
python scripts/train.py --config configs/experiments/gsm8k_3agent.yaml
```

## Project Structure

- `configs/` — Role definitions, graph priors, experiment configs
- `src/models/` — Base model wrapper, compressor, agent
- `src/graph/` — Learnable adjacency matrix, DAG executor
- `src/communication/` — Latent message passing
- `src/losses/` — Task loss + graph regularization
- `src/pipeline/` — Top-level MultiAgentSystem
- `envs/` — Local virtual environments (gitignored)
- `weights/` — Model weights cache (gitignored)
