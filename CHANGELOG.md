# CHANGELOG

## v0.3.0 — 固定拓扑 + Concat 聚合 + Adjacency 监控 (2026-04-09)

### 新增
- **固定拓扑** (`graph.freeze_topology: true`)：冻结 adjacency 的 sigmoid 权重，不参与优化，graph_loss 自动归零
- **Concat 聚合** (`graph.aggregation_mode: "concat"`)：上游 prefix 改为沿序列维度拼接，不再做加权求和
  - 例如 refiner 接收 analyst/critic/verifier 三个 prefix → 输出 `[B, 3*Lp, D]`
  - PrefixProjector 天然支持变长序列，无需额外修改
- **Per-edge adjacency 监控**：每个 optimizer step 记录每条边的 weight、logit、total_grad、task_grad、graph_grad、sigmoid_deriv，输出到 `adjacency_log.json`

### 修改的文件
| 文件 | 变更 |
|------|------|
| `src/communication/aggregator.py` | 重写，支持 `weighted_sum` / `concat` 两种模式 |
| `src/pipeline/multi_agent_system.py` | 添加 `freeze_topology` 冻结逻辑 + `aggregation_mode` 配置传入 |
| `src/cli/train.py` | 添加 `compute_per_edge_adjacency_stats()`；冻结时跳过 adjacency 优化器/梯度同步 |
| `configs/experiments/competition_math_4b_8gpu.yaml` | 添加 `freeze_topology` / `aggregation_mode` 配置 |

### 向后兼容
- 默认 `freeze_topology: false`、`aggregation_mode: "weighted_sum"`，行为与 v0.2.0 一致

---

## v0.2.0 — 基线训练 (commit 9595207, 2026-04-07)

基于 commit 9595207 的 worktree，包含：
- Hierarchical 6-agent DAG (planner→analyst/critic/verifier→refiner→solver)
- Learnable adjacency (sigmoid over logits, init_scale=6.0)
- Weighted-sum prefix aggregation
- LatentCompressor (Q-Former) + PrefixProjector (MLP→per-layer KV cache)
- DDP 8-GPU 训练，AMP bf16
- Competition Math + GSM8K 评测
