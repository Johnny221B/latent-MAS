# Agent Architecture Topologies

This document describes the multi-agent graph topologies used in the arch comparison experiments (`configs/experiments/arch/`). All experiments use Qwen3-4B, trained on GSM8K for 4 epochs.

---

## Overview

Each topology defines:
- **Agents**: ordered list of roles
- **Adjacency prior**: binary matrix `A[i,j] = 1` means agent `i` sends a latent prefix to agent `j`
- **Terminal agent**: the final agent whose output is used for loss / evaluation
- **Parallelizable levels**: agents with no dependencies on each other at the same DAG level are batched into a single forward call by `DAGExecutor`

---

## 1. Sequential (chain_4agent)

**Config**: `configs/graphs/chain_4agent.json`
**Agents**: planner → critic → refiner → solver

```
planner → critic → refiner → solver(terminal)
```

**Adjacency**:
```
         planner  critic  refiner  solver
planner  [  0       1       0       0  ]
critic   [  0       0       1       0  ]
refiner  [  0       0       0       1  ]
solver   [  0       0       0       0  ]
```

Each agent receives only from its immediate predecessor. No parallelism — strictly sequential. Serves as the simplest baseline.

---

## 2. Dense (dense_4agent)

**Config**: `configs/graphs/dense_4agent.json`
**Agents**: planner → critic → refiner → solver

```
planner → critic   → refiner → solver(terminal)
       ↘          ↗ ↘       ↗
        ──────────   ───────
(skip connections: planner→refiner, planner→solver, critic→solver)
```

**Adjacency**:
```
         planner  critic  refiner  solver
planner  [  0       1       1       1  ]
critic   [  0       0       1       1  ]
refiner  [  0       0       0       1  ]
solver   [  0       0       0       0  ]
```

Fully connected DAG — every earlier agent feeds every later agent via skip connections. Solver receives aggregated latent prefixes from all three upstream agents.

---

## 3. Two-Path (two_path_4agent)

**Config**: `configs/graphs/two_path_4agent.json`
**Agents**: planner, analyst, critic, solver

```
planner → analyst ↘
                   solver(terminal)
          critic  ↗
```

**Adjacency**:
```
         planner  analyst  critic  solver
planner  [  0       1        0       0  ]
analyst  [  0       0        0       1  ]
critic   [  0       0        0       1  ]
solver   [  0       0        0       0  ]
```

Two independent paths merge at the solver. `planner → analyst → solver` and `critic → solver` (critic has no upstream). Analyst and critic execute in parallel (same DAG level).

---

## 4. Hierarchical 4-Agent (hierarchical_4agent)

**Config**: `configs/graphs/hierarchical_4agent.json`
**Agents**: planner, critic, refiner, solver

```
planner ↘
         refiner → solver(terminal)
critic  ↗
```

**Adjacency**:
```
         planner  critic  refiner  solver
planner  [  0       0       1       0  ]
critic   [  0       0       1       0  ]
refiner  [  0       0       0       1  ]
solver   [  0       0       0       0  ]
```

Planner and critic run independently in parallel (no connection between them), both feed into refiner, refiner feeds solver. Two DAG levels before the terminal.

---

## 5. Diamond 5-Agent (diamond_5agent)

**Config**: `configs/graphs/diamond_5agent.json`
**Agents**: planner, analyst, critic, refiner, solver

```
         → analyst ↘
planner              refiner → solver(terminal)
         → critic  ↗
```

**Adjacency**:
```
         planner  analyst  critic  refiner  solver
planner  [  0       1        1       0       0  ]
analyst  [  0       0        0       1       0  ]
critic   [  0       0        0       1       0  ]
refiner  [  0       0        0       0       1  ]
solver   [  0       0        0       0       0  ]
```

Classic diamond shape. Planner fans out to analyst and critic (parallel), both merge into refiner, refiner feeds solver. No skip connections. Three sequential levels.

---

## 6. Hierarchical 6-Agent (hierarchical_6agent)

**Config**: `configs/graphs/hierarchical_6agent.json`
**Agents**: planner, analyst, critic, verifier, refiner, solver

```
          → analyst  ↘
planner   → critic    → refiner → solver(terminal)
          → verifier ↗
```

**Adjacency**:
```
          planner  analyst  critic  verifier  refiner  solver
planner   [  0       1        1       1         0       0  ]
analyst   [  0       0        0       0         1       0  ]
critic    [  0       0        0       0         1       0  ]
verifier  [  0       0        0       0         1       0  ]
refiner   [  0       0        0       0         0       1  ]
solver    [  0       0        0       0         0       0  ]
```

Largest topology. Planner fans out to three parallel reviewers (analyst, critic, verifier), all three merge into refiner, refiner feeds solver. Three sequential levels, maximum parallel breadth.

---

## Comparison Summary

| Name | Agents | Levels | Max Parallel Width | Skip Connections |
|------|--------|--------|--------------------|-----------------|
| Sequential | 4 | 4 | 1 | No |
| Dense | 4 | 4 | 1 | Yes (all-to-all) |
| Two-Path | 4 | 3 | 2 | No |
| Hierarchical-4 | 4 | 3 | 2 | No |
| Diamond-5 | 5 | 4 | 2 | No |
| Hierarchical-6 | 6 | 4 | 3 | No |

**Levels**: number of sequential execution steps (DAG depth).
**Max Parallel Width**: maximum number of agents executing concurrently in a single level.
**Skip Connections**: whether any agent receives input from a non-adjacent upstream agent.

---

## Experiment Config

All arch experiments share:
```yaml
model:    Qwen3-4B
task:     gsm8k
epochs:   4
batch_size: 4
gradient_accumulation_steps: 16
lr: 1e-5
use_amp: true
reasoning.steps_per_agent: 40
reasoning.compress_last_k: 40
compressor.num_queries: 16
```

Eval: `inference_mode: chat_with_text`, 4 GPU workers, batch size 1.
Results written to `outputs/arch_gsm8k_4ep/<topology>_eval/eval_results.json`.
