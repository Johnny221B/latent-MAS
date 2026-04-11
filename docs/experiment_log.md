# Experiment Log

All experiments evaluated on GSM8K test set (1319 samples unless noted).

### Precision Notes
- **Model (frozen LLM)**: loaded in float32; with AMP, forward pass runs under bf16 autocast
- **Latent modules (Compressor, PrefixProjector, Adjacency)**: parameters in float32; with AMP, forward bf16, backward float32
- **Without AMP**: everything float32
- **GradScaler**: disabled for bf16 (only used for fp16)

---

## 1. Architecture Comparison — GSM8K, Qwen3-4B, 4 epochs, AMP, no system prompt

| Architecture | Agents | GSM8K | Correct | Output Dir |
|-------------|--------|-------|---------|-----------|
| **Two-path** | 4 | **91.28%** | 1204 | `arch_gsm8k_4ep/two_path_4agent_*` |
| Hierarchical-6 | 6 | 91.13% | 1202 | `arch_gsm8k_4ep/hierarchical_6agent_*` |
| Diamond | 5 | 91.21% | 1203 | `arch_gsm8k_4ep/diamond_5agent_*` |
| Sequential (chain) | 4 | 91.05% | 1201 | `arch_gsm8k_4ep/sequential_4agent_*` |
| Dense (full DAG) | 4 | 90.75% | 1197 | `arch_gsm8k_4ep/dense_4agent_*` |
| Hierarchical-4 | 4 | 90.67% | 1196 | `arch_gsm8k_4ep/hierarchical_4agent_*` |

Config: `batch_size=4, grad_accum=2, lr=1e-5, adj_lr=0.05, steps=40, queries=16, comp_layers=1, AMP=true`
Data: GSM8K train

### Sequential bf16 variants (batch_size=16)

| Variant | GSM8K | Output Dir |
|---------|-------|-----------|
| bf16_20260404_193909 | 91.21% (1203) | `arch_gsm8k_4ep/sequential_4agent_bf16_20260404_193909` |
| bf16_20260404_230425 | 91.05% (1201) | `arch_gsm8k_4ep/sequential_4agent_bf16_20260404_230425` |
| bf16_20260404_131726 | 90.60% (1195) | `arch_gsm8k_4ep/sequential_4agent_bf16_20260404_131726` |
| bf16_20260404_170948 | 90.45% (1193) | `arch_gsm8k_4ep/sequential_4agent_bf16_20260404_170948` |

## 2. Multi-Benchmark Evaluation — Sequential 4-agent, AM-R1 data, 2 epochs, bf16

Model: Qwen3-4B, `batch_size=4, grad_accum=2, lr=1e-5, AMP=true`
Data: AM DeepSeek R1 distilled (source: am-0309)

| Benchmark | Score | Samples | Output Dir |
|-----------|-------|---------|-----------|
| **GSM8K** | **91.05%** | 1319 | `arch_comparison/am_sequential_4agent_2ep_bf16_*/gsm8k_eval` |
| **ARC-Easy** | **95.66%** | 2376 | `arch_comparison/am_sequential_4agent_2ep_bf16_*/arc_easy` |
| **ARC-Challenge** | **90.61%** | 1172 | `arch_comparison/am_sequential_4agent_2ep_bf16_*/arc_challenge` |
| **MATH-500** | **53.91%** | 500 | `arch_comparison/am_sequential_4agent_2ep_bf16_*/math500` |
| **HumanEval** | **45.45%** (pass@1) | 66 | `arch_comparison/am_sequential_4agent_2ep_bf16_*/humaneval` |

## 3. Architecture Comparison — 2 epochs (earlier run)

| Architecture | Agents | Data | GSM8K | Output Dir |
|-------------|--------|------|-------|-----------|
| Sequential (chain) | 4 | AM-R1 | 87.26% (1151) | `arch_comparison/sequential_4agent_*` |
| Dense (full DAG) | 4 | AM-R1 | 87.19% (1150) | `arch_comparison/dense_4agent_*` |

## 4. Hierarchical 6-agent (方案C) — Model Size Comparison

Graph: `planner → (analyst, critic, verifier) → refiner → solver`
Data: AM-R1 distilled, AMP

| Model | Epochs | BS | Model Precision | Latent Precision | GSM8K | Output Dir |
|-------|--------|----|----------------|-----------------|-------|-----------|
| **Qwen3-4B** | 2 | 4 | fp32 + bf16 fwd | fp32 + bf16 fwd | **87.72%** (1157) | `hier6/am_deepseek_r1_4b_amp_*` |
| Qwen3-8B | 2 | 4 | fp32 + bf16 fwd | fp32 + bf16 fwd | 79.98% (1055) | `hier6/am_deepseek_r1_8b_amp_*` |
| Qwen3-1.7B | 4 | 8 | fp32 + bf16 fwd | fp32 + bf16 fwd | 77.71% (1025) | `hier6/am_deepseek_r1_1.7b_amp_4ep_*` |
| Qwen3-1.7B | 2 | 8 | fp32 + bf16 fwd | fp32 + bf16 fwd | 77.56% (1023) | `hier6/am_deepseek_r1_1.7b_amp_*` |
| 4B+1.7B 异构 | 2 | 4 | fp32 + bf16 fwd | fp32 + bf16 fwd | eval pending | `hier6/am_deepseek_r1_4b_1.7b_amp_*` |

## 5. Graph v2 — Sequential 4-agent Chain

Graph: `planner → critic → refiner → solver`

| Model | Data | Model Precision | Latent Precision | GSM8K | Output Dir |
|-------|------|----------------|-----------------|-------|-----------|
| **Qwen3-8B** | AM-R1 | fp32 | fp32 | **90.45%** (1193) | `graph_v2/am_deepseek_r1_8b_*` |
| Qwen3-8B | GSM8K | fp32 | fp32 | 86.96% (1147) | `graph_v2/gsm8k_qwen3-8b_*` |
| Qwen3-8B | GSM8K | fp32 | fp32 | 85.22% (1124) | text-only eval |
| Qwen3-4B | GSM8K | fp32 | fp32 | 86.43% (1140) | `graph_v2/gsm8k_qwen3-4b_*` |
| Qwen3-4B | GSM8K | fp32 | fp32 | 83.78% (1105) | text-only eval |

## 6. 1.7B Hyperparameter Sweep (sequential 4-agent, GSM8K, 2 epochs)

All: Qwen3-1.7B, fp32, no AMP

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

## 7. 4B Hyperparameter Sweep (sequential 5-agent, GSM8K)

All: Qwen3-4B, fp32, no AMP

| Epochs | Steps | Queries | Comp Layers | GSM8K | Output Dir |
|--------|-------|---------|------------|-------|-----------|
| ep5 | s40 | q16 | cl1 | 86.13% (1136) | `gsm8k-4b-ep5-s40-q16-cl1_*` |
| ep2 | s40 | q16 | cl1 | 86.05% (1135) | `gsm8k-4b-ep2-s40-q16-cl1_*` |
| ep5 | s40 | q16 | cl2 | 85.90% (1133) | `gsm8k-4b-ep5-s40-q16-cl2_*` |
| ep5 | s40 | q32 | cl1 | 85.60% (1129) | `gsm8k-4b-ep5-s40-q32-cl1_*` |

## 8. Earlier Experiments (default 5-agent/3-agent, GSM8K)

| Date | Model | Graph | Model Precision | Latent Precision | GSM8K | Output Dir |
|------|-------|-------|----------------|-----------------|-------|-----------|
| 04-02 | Qwen3-4B | revised prompt | fp32 | fp32 | 87.19% (1150) | `gsm8k_qwen3-4b_20260402_*` |
| 04-01 | Qwen3-8B | 5-agent | fp32 | fp32 | 86.66% (1143) | `gsm8k_qwen3-8b_20260401_054040` |
| 04-01 | Qwen3-4B | 5-agent | fp32 | fp32 | 85.97% (1134) | `gsm8k_qwen3-4b_20260401_044523` |
| 03-31 | Qwen3-4B | 5-agent | fp32 | fp32 | 86.20% (1137) | `gsm8k_qwen3-4b_20260331_*` |
| 03-29 | Qwen3-4B | 5-agent | fp32 | fp32 | 79.30% (1046) | `gsm8k_qwen3-4b_20260329_*` |
| 03-28 | Qwen3-4B | 5-agent | fp32 | fp32 | 74.75% (986) | `gsm8k_qwen3-4b_20260328_*` |
| 03-23 | Qwen3-8B | 3-agent | fp32 | fp32 | 78.92% (1041) | `gsm8k_qwen3-8b_20260323_*` |

---

## Training Optimizations Benchmark (Qwen3-4B, 2×B200, hier6)

| Config | Avg fwd+bwd | Memory | Speedup |
|--------|------------|--------|---------|
| Baseline (sequential, fp32) | 8.5s | 144 GB | 1.0x |
| + Batched agents | 8.5s | 112 GB | 1.0x fwd, -22% mem |
| + AMP (bf16) | **4.4s** | **85 GB** | **1.93x** |

## Graph Topologies

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
| 28812957 | Sequential 4-agent, AM-R1, 2ep, bf16, think=true + 6 evals | submitted |

---

## Auto-tracked Results

_Auto-updated by `scripts/monitor_experiments.sh`_

| Date | Model | Graph | AMP | Epochs | BS | GSM8K | Output Dir |
|------|-------|-------|-----|--------|----|-------|-----------|
| 04-04 | Qwen3-4B | chain_4agent | True | ? | 4 | **92.50%** (37/40) | `sequential_4agent_20260404_101205` |
| 04-04 | Qwen3-4B | two_path_4agent | True | ? | 4 | **100.00%** (8/8) | `two_path_4agent_20260404_102104` |
| 04-04 | Qwen3-4B | hierarchical_6agent | True | ? | 4 | **93.75%** (15/16) | `hierarchical_6agent_20260404_091712` |
| 04-04 | Qwen3-4B | chain_4agent | True | ? | 16 | **92.50%** (37/40) | `sequential_4agent_bf16_20260404_131726` |
| 04-04 | Qwen3-4B | chain_4agent | True | ? | 16 | **94.44%** (34/36) | `sequential_4agent_bf16_20260404_170948` |
| 04-04 | Qwen3-4B | chain_4agent | True | ? | 16 | **100.00%** (4/4) | `sequential_4agent_bf16_20260404_193909` |
| 04-04 | Qwen3-4B | chain_4agent | True | ? | 16 | **93.18%** (41/44) | `sequential_4agent_bf16_20260404_230425` |
| 04-05 | Qwen3-1.7B+Qwen3-4B | hierarchical_6agent | True | ? | 16 | **92.71%** (356/384) | `hier6_4b_1.7b_bf16_20260404_152531` |
| 04-05 | Qwen3-4B | chain_4agent | True | ? | 4 | **93.12%** (298/320) | `am_sequential_4agent_2ep_bf16_20260405_035543` |
| 04-05 | Qwen3-4B | chain_4agent | True | ? | 4 | **96.88%** (62/64) | `am_sequential_4agent_2ep_bf16_think_20260405_124911` |
| 04-05 | Qwen3-4B | chain_4agent | True | ? | 8 | **95.31%** (61/64) | `competition_math_sequential_4agent_1ep_bf16_think_20260405_134422` |
| 04-06 | Qwen3-4B | chain_4agent | True | ? | 4 | **50.00%** (6/12) | `sequential_4agent_bf16_20260405_213221` |
| 04-06 | Qwen3-1.7B+Qwen3-4B | hierarchical_6agent | True | ? | 8 | **95.31%** (61/64) | `competition_math_hier6_4b_1.7b_4ep_bf16_think_20260406_043621` |
| 04-06 | Qwen3-1.7B+Qwen3-4B | hierarchical_6agent | True | ? | 8 | **96.88%** (31/32) | `competition_math_hier6_4b_1.7b_2ep_bf16_think_lowreg_20260406_085643` |
| 04-06 | Qwen3-4B | chain_4agent | True | ? | 8 | **100.00%** (4/4) | `competition_math_seq4_2ep_bf16_think_lowreg_20260406_200645` |
| 04-06 | Qwen3-4B | hierarchical_6agent | True | ? | 4 | **100.00%** (12/12) | `hierarchical_6agent_bf16_20260406_201542` |
| 04-07 | Qwen3-4B | hierarchical_6agent | True | ? | 4 | **93.75%** (15/16) | `smoke` |
| 04-07 | Qwen3-4B | hierarchical_6agent | True | ? | 4 | **100.00%** (8/8) | `dev_20260407_052959` |
| 04-07 | Qwen3-4B | hierarchical_6agent | True | ? | 8 | **93.75%** (15/16) | `hierarchical_6a_4b_20260407_055204` |
| 04-07 | Qwen3-4B | hierarchical_6agent | True | ? | 8 | **93.75%** (15/16) | `hierarchical_6a_4b_20260407_064046` |
