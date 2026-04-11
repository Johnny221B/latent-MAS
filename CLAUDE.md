# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Latent Multi-Agent System (LatentMAS) — a trainable framework where multiple frozen LLM agents collaborate via **latent-space communication** (compressed hidden states as per-layer KV-cache prefixes) instead of text. Three components are trainable: the **LatentCompressor** (cross-attention Q-Former that compresses agent hidden states to fixed-length prefixes), the **PrefixProjector** (MLP that projects compressed prefixes to per-layer KV cache, following Prefix-Tuning reparameterization), and the **LearnableAdjacency** (sigmoid-parameterized DAG defining which agents communicate).

Key recent additions:
- **Heterogeneous models**: different agents can use different-sized LLMs (e.g. 4B for planner/solver, 1.7B for middle-layer agents), with `HiddenProjection` for dimension alignment
- **Batched agent execution**: independent agents at the same DAG level are auto-detected and batched into a single forward call
- **Mixed precision (AMP)**: optional bf16 autocast via `training.use_amp: true`
- **LR scheduler**: optional linear warmup + cosine decay via `training.warmup_steps`
- **Multiple graph topologies**: sequential, dense, diamond, hierarchical, two-path

## Commands

### Setup
```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### Training
```bash
bash scripts/train.sh --config configs/experiments/gsm8k_5agent.yaml
```
Training uses `torchrun` → `src/cli/train.py`. Output goes to `outputs/<timestamp>/`.

### Evaluation
```bash
CKPT_FOLDER=outputs/<checkpoint_dir> bash scripts/evaluate.sh     # ours
bash scripts/eval_single.sh                                        # single-model baseline (vLLM)
bash scripts/eval_paper.sh                                         # paper LatentMAS baseline
```

### SLURM
```bash
sbatch scripts/train/<experiment>.slurm    # training + eval
sbatch scripts/arch/<architecture>.slurm   # architecture comparison
sbatch scripts/eval/<eval_task>.slurm      # eval only
```

## Architecture

### Data Flow (Training)
```
Tokenized batch → MultiAgentSystem.forward()
  → DAGExecutor groups agents into parallelizable levels from adjacency matrix:
      For each level:
        If multiple independent non-terminal agents → batched execution (single forward)
        Otherwise → sequential execution:
          1. MessageAggregator: weighted sum of upstream prefixes (using adjacency weights)
          2. PrefixProjector: prefix → per-layer KV cache
          3. Agent.reason(): m-step latent reasoning producing hidden states
             (if heterogeneous: HiddenProjection aligns dimensions)
          4. LatentCompressor: hidden states → fixed-length prefix
  → Terminal agent logits → TaskLoss (CE on answer tokens)
  → LearnableAdjacency → GraphLoss (edge regularization)
  → Backprop updates LatentCompressor + PrefixProjector + LearnableAdjacency
    (+ HiddenProjection if heterogeneous)
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
| **HiddenProjection** | `src/models/compressor.py` | Linear + LayerNorm projection for cross-model dimension alignment |
| **DAGExecutor** | `src/graph/dag_executor.py` | Level-grouped agent execution; batches independent agents at same DAG level |
| **MessageAggregator** | `src/communication/aggregator.py` | z_j = Σ A[i,j] * P_i; gradients flow into adjacency |
| **TaskLoss / GraphLoss** | `src/losses/` | CE on answer tokens; edge-change penalties (λ_add, λ_drop, λ_sparse) |
| **Datasets** | `src/data/` | GSM8K, ARC, HumanEval, Competition Math, AM DeepSeek R1 — all normalized to {question, answer} |

### Configuration
- **Experiment configs**: `configs/experiments/` — organized into subdirectories:
  - `arch/` — architecture comparison experiments (sequential, dense, diamond, hierarchical, two-path)
  - `hier6/` — hierarchical 6-agent experiments (1.7B, 4B, 8B, heterogeneous, AMP variants)
  - `benchmark/` — forward pass benchmarks
  - `sweep/`, `sweep_1.7b/` — hyperparameter sweeps
  - `legacy/` — older experiments
- **Graph topology**: `configs/graphs/*.json` — agents list, adjacency_prior matrix, execution_order, terminal_agent_index
  - `chain_4agent.json` — sequential: planner→critic→refiner→solver
  - `dense_4agent.json` — fully connected DAG with skip connections
  - `two_path_4agent.json` — two independent paths merging at solver
  - `hierarchical_4agent.json` — (planner,critic)→refiner→solver
  - `diamond_5agent.json` — planner→(analyst,critic)→refiner→solver
  - `hierarchical_6agent.json` — planner→(analyst,critic,verifier)→refiner→solver
- **Role definitions**: `configs/roles/*.json` — role_name, role_prompt, reasoning_steps per agent
- **Eval configs**: `configs/eval/*.yaml`

### Heterogeneous Model Config
```yaml
model:
  name: "Qwen/Qwen3-4B"          # default / fallback
  models:
    primary: "Qwen/Qwen3-4B"
    secondary: "Qwen/Qwen3-1.7B"
  agent_models: ["primary", "secondary", "secondary", "secondary", "primary", "primary"]
```

### Training Optimizations

## Environment & Debugging

### Python Environment
- Use `uv` for all package management
- Virtual environment: `.venv` (Python 3.11)
- Setup: `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`

### GPU Node (for debugging and execution)
- **All program execution and debugging must go through SSH on the GPU node**
- Connect: `ssh root@10.100.38.15 -p 24012`
- Hostname: lg-cmc-b7r402-e03u16-h200-001221
- GPUs: 8x NVIDIA H200 (143GB each)
- **CRITICAL: The GPU node has NO internet/network access — cannot pip install, download models, or fetch remote resources. All files and dependencies must already be present on the node.**

### Code Editing
- Edit code in the worktree: `.claude/worktrees/prev-commit/`
- The worktree is based on commit `9595207`

## Documentation Policy (from AGENTS.md)

- `docs/training_pipeline.md` is the canonical description of the train/eval pipeline
- `docs/method.md`, `docs/agent_workflow.md`, `docs/prompt_flow.md` must stay aligned with runtime behavior
- After changes affecting train/eval flow, config semantics, output artifacts, or prompt behavior — update the corresponding docs in the same session
- When docs and code diverge, fix the docs immediately
