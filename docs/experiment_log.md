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
| **Sequential (chain)** | 4 | **91.28%** | 1204 | `arch_comparison/sequential_4agent_no_sp_*` |
| Diamond | 5 | 91.21% | 1203 | `arch_gsm8k_4ep/diamond_5agent_*` |
| Dense (full DAG) | 4 | 90.75% | 1197 | `arch_gsm8k_4ep/dense_4agent_*` |
| Hierarchical-4 | 4 | 90.67% | 1196 | `arch_gsm8k_4ep/hierarchical_4agent_*` |
| Two-path | 4 | pending | — | — |
| Hierarchical-6 | 6 | pending | — | — |

Config: `batch_size=4, grad_accum=16, lr=1e-5, adj_lr=0.01, steps=40, queries=16, comp_layers=1`
Data: AM-R1 distilled (sequential_no_sp) / GSM8K (others)

## 2. Architecture Comparison — 2 epochs (earlier run)

| Architecture | Agents | Data | GSM8K | Output Dir |
|-------------|--------|------|-------|-----------|
| Sequential (chain) | 4 | AM-R1 | 87.26% (1151) | `arch_comparison/sequential_4agent_*` |
| Dense (full DAG) | 4 | AM-R1 | 87.19% (1150) | `arch_comparison/dense_4agent_*` |

## 3. Hierarchical 6-agent (方案C) — Model Size Comparison

Graph: `planner → (analyst, critic, verifier) → refiner → solver`
Data: AM-R1 distilled, AMP

| Model | Epochs | BS | Model Precision | Latent Precision | GSM8K | Output Dir |
|-------|--------|----|----------------|-----------------|-------|-----------|
| **Qwen3-4B** | 2 | 4 | fp32 + bf16 fwd | fp32 + bf16 fwd | **87.72%** (1157) | `hier6/am_deepseek_r1_4b_amp_*` |
| Qwen3-8B | 2 | 4 | fp32 + bf16 fwd | fp32 + bf16 fwd | 79.98% (1055) | `hier6/am_deepseek_r1_8b_amp_*` |
| Qwen3-1.7B | 4 | 8 | fp32 + bf16 fwd | fp32 + bf16 fwd | 77.71% (1025) | `hier6/am_deepseek_r1_1.7b_amp_4ep_*` |
| Qwen3-1.7B | 2 | 8 | fp32 + bf16 fwd | fp32 + bf16 fwd | 77.56% (1023) | `hier6/am_deepseek_r1_1.7b_amp_*` |
| 4B+1.7B 异构 | 2 | 4 | fp32 + bf16 fwd | fp32 + bf16 fwd | eval pending | `hier6/am_deepseek_r1_4b_1.7b_amp_*` |

## 4. Graph v2 — Sequential 4-agent Chain

Graph: `planner → critic → refiner → solver`

| Model | Data | Model Precision | Latent Precision | GSM8K | Output Dir |
|-------|------|----------------|-----------------|-------|-----------|
| **Qwen3-8B** | AM-R1 | fp32 | fp32 | **90.45%** (1193) | `graph_v2/am_deepseek_r1_8b_*` |
| Qwen3-8B | GSM8K | fp32 | fp32 | 86.96% (1147) | `graph_v2/gsm8k_qwen3-8b_*` |
| Qwen3-8B | GSM8K | fp32 | fp32 | 85.22% (1124) | text-only eval |
| Qwen3-4B | GSM8K | fp32 | fp32 | 86.43% (1140) | `graph_v2/gsm8k_qwen3-4b_*` |
| Qwen3-4B | GSM8K | fp32 | fp32 | 83.78% (1105) | text-only eval |

## 5. 1.7B Hyperparameter Sweep (sequential 4-agent, GSM8K, 2 epochs)

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

## 6. 4B Hyperparameter Sweep (sequential 5-agent, GSM8K)

All: Qwen3-4B, fp32, no AMP

| Epochs | Steps | Queries | Comp Layers | GSM8K | Output Dir |
|--------|-------|---------|------------|-------|-----------|
| ep5 | s40 | q16 | cl1 | 86.13% (1136) | `gsm8k-4b-ep5-s40-q16-cl1_*` |
| ep2 | s40 | q16 | cl1 | 86.05% (1135) | `gsm8k-4b-ep2-s40-q16-cl1_*` |
| ep5 | s40 | q16 | cl2 | 85.90% (1133) | `gsm8k-4b-ep5-s40-q16-cl2_*` |
| ep5 | s40 | q32 | cl1 | 85.60% (1129) | `gsm8k-4b-ep5-s40-q32-cl1_*` |

## 7. Earlier Experiments (default 5-agent/3-agent, GSM8K)

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
| 28736665-28736670 | arch comparison 4B × 6 archs (GSM8K, 4ep) | running |
| 28736665-28736670 (8B) | arch comparison 8B × 6 archs (GSM8K, 4ep) | pending submit |
| 28734985 | eval 4B+1.7B heterogeneous | submitted |

---

## Auto-tracked Results

_Auto-updated by `scripts/monitor_experiments.sh`_

| Date | Model | Graph | AMP | Epochs | BS | GSM8K | Output Dir |
|------|-------|-------|-----|--------|----|-------|-----------|
| 04-04 | Qwen3-4B | chain_4agent | True | ? | 4 | **92.50%** (37/40) | `sequential_4agent_20260404_101205` |
| 04-04 | Qwen3-4B | two_path_4agent | True | ? | 4 | **100.00%** (8/8) | `two_path_4agent_20260404_102104` |
| 04-04 | Qwen3-4B | hierarchical_6agent | True | ? | 4 | **93.75%** (15/16) | `hierarchical_6agent_20260404_091712` |
| 04-04 | Qwen3-4B | chain_4agent | True | ? | 16 | **92.50%** (37/40) | `sequential_4agent_bf16_20260404_131726` |
