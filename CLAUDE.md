# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Latent Multi-Agent System (LatentMAS) — a trainable framework where multiple frozen LLM agents collaborate via **latent-space communication** (compressed hidden states as per-layer KV-cache prefixes) instead of text. Three components are trainable: the **LatentCompressor** (cross-attention Q-Former that compresses agent hidden states to fixed-length prefixes), the **PrefixProjector** (MLP that projects compressed prefixes to per-layer KV cache, following Prefix-Tuning reparameterization), and the **LearnableAdjacency** (sigmoid-parameterized DAG defining which agents communicate).

## Commands

### Setup
```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### Training
```bash
bash scripts/train.sh                                              # default (2 GPU)
NPROC_PER_NODE=2 bash scripts/train.sh --config configs/experiments/gsm8k_5agent.yaml
```
Training uses `torchrun` → `src/cli/train.py`. Output goes to `outputs/<timestamp>/`.

### Evaluation
```bash
CKPT_FOLDER=outputs/<checkpoint_dir> bash scripts/evaluate.sh     # ours
bash scripts/eval_single.sh                                        # single-model baseline (vLLM)
bash scripts/eval_paper.sh                                         # paper LatentMAS baseline
```

### Tests
```bash
pytest tests/                          # all tests
pytest tests/test_e2e.py -v           # end-to-end training
pytest tests/test_train.py            # training loop
pytest tests/test_evaluate_streaming.py  # eval pipeline
```

### SLURM
```bash
sbatch scripts/job.slurm
```

## Architecture

### Data Flow (Training)
```
Tokenized batch → MultiAgentSystem.forward()
  → DAGExecutor processes agents in topological order:
      For each agent:
        1. MessageAggregator: weighted sum of upstream prefixes (using adjacency weights)
        2. PrefixProjector: prefix → per-layer KV cache
        3. Agent.reason(): m-step generation producing hidden states
        4. LatentCompressor: hidden states → fixed-length prefix
  → Terminal agent logits → TaskLoss (CE on answer tokens)
  → LearnableAdjacency → GraphLoss (edge regularization)
  → Backprop updates only LatentCompressor + PrefixProjector + LearnableAdjacency
```

### Key Modules
| Module | Location | Role |
|--------|----------|------|
| **MultiAgentSystem** | `src/pipeline/multi_agent_system.py` | Top-level nn.Module; owns all trainable params, orchestrates forward pass |
| **BaseModelWrapper** | `src/models/base_model.py` | Frozen HF causal LM (e.g. Qwen3-4B) + tokenizer |
| **Agent** | `src/models/agent.py` | Role-aware wrapper; performs m-step latent reasoning, does NOT own model weights |
| **LatentCompressor** | `src/models/compressor.py` | Cross-attention compressor: variable-length hidden states → Lp fixed-length prefix |
| **PrefixProjector** | `src/models/compressor.py` | Maps compressed prefix → per-layer KV cache (MLP reparameterization) |
| **LearnableAdjacency** | `src/graph/adjacency.py` | Sigmoid over raw logits W; DAG constraint via upper-triangular mask |
| **DAGExecutor** | `src/graph/dag_executor.py` | Topological-order agent execution with prefix aggregation |
| **MessageAggregator** | `src/communication/aggregator.py` | z_j = Σ A[i,j] * P_i; gradients flow into adjacency |
| **TaskLoss / GraphLoss** | `src/losses/` | CE on answer tokens; edge-change penalties (λ_add, λ_drop, λ_sparse) |
| **Datasets** | `src/data/` | GSM8K, ARC, HumanEval, Competition Math, AM DeepSeek R1 — all normalized to {question, answer} |

### Configuration
- **Experiment configs**: `configs/experiments/*.yaml` — model, graph, compressor, training, reasoning params
- **Graph topology**: `configs/graphs/*.json` — agents list, adjacency_prior matrix, execution_order, terminal_agent_index
- **Role definitions**: `configs/roles/*.json` — role_name, system_prompt, reasoning_steps per agent
- **Eval configs**: `configs/eval/*.yaml`

## Documentation Policy (from AGENTS.md)

- `docs/training_pipeline.md` is the canonical description of the train/eval pipeline
- `docs/method.md`, `docs/agent_workflow.md`, `docs/prompt_flow.md` must stay aligned with runtime behavior
- After changes affecting train/eval flow, config semantics, output artifacts, or prompt behavior — update the corresponding docs in the same session
- When docs and code diverge, fix the docs immediately
