# Experiment Log

All experiments evaluated on GSM8K test set (1319 samples unless noted).

---

## 1. Hierarchical 6-agent (方案C) — Model Size & Training Comparison

Graph: `planner → (analyst, critic, verifier) → refiner → solver`

| Date | Model | AMP | Epochs | BS | GSM8K | Output Dir |
|------|-------|-----|--------|----|-------|-----------|
| 04-03 | **Qwen3-4B** | Yes | 2 | 4 | **87.72%** (1157) | `hier6/am_deepseek_r1_4b_amp_*` |
| 04-03 | Qwen3-8B | Yes | 2 | 4 | 79.98% (1055) | `hier6/am_deepseek_r1_8b_amp_*` |
| 04-03 | Qwen3-1.7B | Yes | 4 | 8 | 77.71% (1025) | `hier6/am_deepseek_r1_1.7b_amp_4ep_*` |
| 04-03 | Qwen3-1.7B | Yes | 2 | 8 | 77.56% (1023) | `hier6/am_deepseek_r1_1.7b_amp_*` |
| 04-03 | 4B+1.7B 异构 | Yes | 2 | 4 | pending | `hier6/am_deepseek_r1_4b_1.7b_amp_*` |

## 2. Architecture Comparison (Qwen3-4B, AMP, 2 epochs)

| Date | Architecture | Agents | GSM8K | Output Dir |
|------|-------------|--------|-------|-----------|
| 04-03 | Dense (full DAG + skip) | 4 | **87.19%** (1150) | `arch_comparison/dense_4agent_*` |
| 04-04 | Sequential (chain) | 4 | 89.02% (235/264)* | `arch_comparison/sequential_4agent_*` |
| 04-04 | Sequential (no SP) | 4 | pending | — |
| — | Two-path | 4 | pending | — |
| — | Hierarchical-4 | 4 | pending | — |
| — | Diamond-5 | 5 | pending | — |
| — | Hierarchical-6 | 6 | 87.72% | (same as hier6 4B above) |

*Sequential eval only ran on 264 samples (partial), not full 1319.

## 3. Graph v2 — Sequential 4-agent Chain (earlier experiments)

Graph: `planner → critic → refiner → solver`, data: MATH_metamathQA

| Date | Model | Data | GSM8K | Notes | Output Dir |
|------|-------|------|-------|-------|-----------|
| 04-02 | **Qwen3-8B** | AM-R1 | **90.45%** (1193) | chain, 8k seq | `graph_v2/am_deepseek_r1_8b_*` |
| 04-02 | Qwen3-8B | GSM8K | 86.96% (1147) | chain | `graph_v2/gsm8k_qwen3-8b_*` |
| 04-02 | Qwen3-8B | GSM8K | 85.22% (1124) | text-only eval | `graph_v2/gsm8k_qwen3-8b_*/gsm8k_eval_text` |
| 04-01 | Qwen3-4B | GSM8K | 86.43% (1140) | chain | `graph_v2/gsm8k_qwen3-4b_*` |
| 04-01 | Qwen3-4B | GSM8K | 83.78% (1105) | text-only eval | `graph_v2/gsm8k_qwen3-4b_*/gsm8k_eval_text_only` |

## 4. Graph v2 — 1.7B Hyperparameter Sweep (sequential 4-agent chain)

All: Qwen3-1.7B, GSM8K data, chain graph, 2 epochs

| Steps | Queries | Comp Layers | GSM8K | Output Dir |
|-------|---------|------------|-------|-----------|
| s40 | q16 | cl1 | **76.57%** (1010) | `graph_v2/1.7b-q16-s40-cl1_*` |
| s40 | q16 | cl2 | 76.57% (1010) | `graph_v2/1.7b-q16-s40-cl2_*` |
| s20 | q16 | cl1 | 76.27% (1006) | `graph_v2/1.7b-q16-s20-cl1_*` |
| s60 | q16 | cl1 | 75.97% (1002) | `graph_v2/1.7b-q16-s60-cl1_*` |
| s40 | q8 | cl1 | 76.19% (1005) | `graph_v2/1.7b-q8-s40-cl1_*` |
| s20 | q8 | cl1 | 76.12% (1004) | `graph_v2/1.7b-q8-s20-cl1_*` |
| s40 | q32 | cl1 | 74.15% (978) | `graph_v2/1.7b-q32-s40-cl1_*` |
| s60 | q32 | cl2 | 74.07% (977) | `graph_v2/1.7b-q32-s60-cl2_*` |

## 5. 4B Hyperparameter Sweep (sequential 5-agent, GSM8K data)

All: Qwen3-4B, GSM8K data, default 5-agent graph

| Epochs | Steps | Queries | Comp Layers | GSM8K | Output Dir |
|--------|-------|---------|------------|-------|-----------|
| ep5 | s40 | q16 | cl1 | 86.13% (1136) | `gsm8k-4b-ep5-s40-q16-cl1_*` |
| ep2 | s40 | q16 | cl1 | 86.05% (1135) | `gsm8k-4b-ep2-s40-q16-cl1_*` |
| ep5 | s40 | q16 | cl2 | 85.90% (1133) | `gsm8k-4b-ep5-s40-q16-cl2_*` |
| ep5 | s40 | q32 | cl1 | 85.60% (1129) | `gsm8k-4b-ep5-s40-q32-cl1_*` |

## 6. Earlier Experiments (default 5-agent/3-agent, GSM8K data)

| Date | Model | Graph | GSM8K | Output Dir |
|------|-------|-------|-------|-----------|
| 04-02 | Qwen3-4B | revised prompt | 87.19% (1150) | `gsm8k_qwen3-4b_20260402_*` |
| 04-01 | Qwen3-8B | 5-agent | 86.66% (1143) | `gsm8k_qwen3-8b_20260401_054040` |
| 04-01 | Qwen3-4B | 5-agent | 85.97% (1134) | `gsm8k_qwen3-4b_20260401_044523` |
| 04-01 | Qwen3-4B | graph v1 | 87.00% (87/100) | `graph/gsm8k_qwen3-4b_20260401_*` |
| 03-31 | Qwen3-4B | 5-agent | 86.20% (1137) | `gsm8k_qwen3-4b_20260331_010342` |
| 03-31 | Qwen3-4B | with-prefix eval | 79.00% (1042) | `gsm8k_qwen3-4b_20260331_*/gsm8k-with-prefix` |
| 03-29 | Qwen3-4B | 5-agent | 79.30% (1046) | `gsm8k_qwen3-4b_20260329_*` |
| 03-28 | Qwen3-4B | 5-agent | 74.75% (986) | `gsm8k_qwen3-4b_20260328_*` |
| 03-25 | Qwen3-4B | early | 7.88% (104) | `gsm8k_qwen3-4b_20260325_*` |
| 03-23 | Qwen3-8B | 3-agent | 78.92% (1041) | `gsm8k_qwen3-8b_20260323_042146` |
| 03-23 | Qwen3-8B | competition_math | 75.28% (993) | `competition_math_qwen3-8b_*` |
| 03-19 | Qwen3-8B | early | 72.00% (72/100) | `gsm8k_qwen3-8b_20260319_*` |

---

## Training Optimizations Benchmark (Qwen3-4B, 2×B200, hier6)

| Config | Avg fwd+bwd | Memory | Speedup |
|--------|------------|--------|---------|
| Baseline (sequential, fp32) | 8.5s | 144 GB | 1.0x |
| + Batched agents | 8.5s | 112 GB | 1.0x fwd, -22% mem |
| + AMP (bf16) | **4.4s** | **85 GB** | **1.93x** |

## Graph Topologies Tested

```
A: Sequential        planner → critic → refiner → solver
B: Dense             planner → critic → refiner → solver (+ all skip connections)
C: Two-path          planner → analyst → solver ← critic
D: Hierarchical-4    (planner, critic) → refiner → solver
E: Diamond-5         planner → (analyst, critic) → refiner → solver
F: Hierarchical-6    planner → (analyst, critic, verifier) → refiner → solver
```

## Active / Pending Jobs

| Job ID | Task | Status |
|--------|------|--------|
| 28720532 | 4B+1.7B heterogeneous hier6 | running |
| 28729106 | arch-sequential-4agent (no SP) | running |
| 28731986 | eval sequential-4agent (full) | submitted |
| pending | arch: two_path, hierarchical_4, diamond_5, hierarchical_6 | waiting |
