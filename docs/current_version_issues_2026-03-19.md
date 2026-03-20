# 当前版本程序问题与修复跟踪

这份文档专门记录当前版本程序中已经确认的问题、当前修复状态、以及仍在迭代中的项。它和 `docs/training_pipeline_risks.md` 的区别是：

- `training_pipeline_risks.md` 更偏分析和风险判断
- 本文档更偏当前版本的工程状态和修复跟踪

## 已修复

### 1. 消息聚合公式与 method 不一致

- 文件：`src/communication/aggregator.py`
- 旧行为：对加权和再除以入边权重和，单入边时会弱化边权的任务梯度
- 新行为：改为直接加权求和，不再做归一化
- 当前状态：已修复，单测已覆盖

### 2. `reasoning_steps` 默认值重复赋值

- 文件：`src/models/agent.py`
- 旧行为：初始化里写了两次 `self.reasoning_steps = role_config.get(...)`
- 新行为：保留单一默认值定义
- 当前状态：已修复

### 3. prefix 推理路径只适合 batch size = 1

- 文件：`src/models/agent.py`
- 旧行为：`next_token.item()`、单样本 `decode()`、固定形状 mask 都默认单样本
- 新行为：prefix 分支和无 prefix 分支都支持 batch 维，并对多样本返回批量 metadata
- 当前状态：已修复，单测已覆盖

### 4. DAG 执行器强依赖 agent index 就是拓扑序

- 文件：`src/graph/dag_executor.py`、`src/pipeline/multi_agent_system.py`
- 旧行为：默认只遍历 `range(n)`，且默认最后一个 agent 是 terminal
- 新行为：支持 graph config 中显式 `execution_order`，并显式传递 `terminal_agent_index`
- 当前状态：已修复，单测已覆盖

### 5. 评测脚本缺少 split 维度和时延聚合指标

- 文件：`src/cli/evaluate.py`
- 新增能力：
  - `--split train|test`
  - `avg_sample_seconds`
  - `avg_generated_tokens`
  - `avg_tokens_per_second`
- 当前状态：已修复，兼容旧调用

## 进行中

### 6. 64/64 小样本目标的双卡高显存训练与评测

- 当前目标：
  - 用 `64` 条训练样本训练
  - 在 `64` 条 train / `64` 条 test 上评测
  - 区分 same-split probe 和 held-out generalization 两类指标
- 当前策略：
  - 双卡 DDP 训练
  - 训练后直接做双卡 live eval
  - 不落 checkpoint 文件，只保留轻量日志与结果 JSON
- 当前状态：已完成 3 轮实验，结果见 `docs/probe64_experiment_log_2026-03-19.md`

## 仍未解决 / 需继续观察

### 7. 当前实现仍然只训练 communication layer

- 文件：`src/pipeline/multi_agent_system.py`
- 现状：`get_trainable_params()` 仍只返回 `compressor + adjacency`
- 含义：method 需要和代码保持一致，不应宣称“联合训练所有 agent 的 latent reasoning”
- 当前状态：文档已收紧，算法层是否扩展仍取决于当前版本能否达标

### 8. DDP 仍启用了 `find_unused_parameters=True`

- 文件：`src/cli/train.py`
- 现象：双卡训练时 PyTorch 发出额外遍历 autograd 图的 warning
- 风险：功能上不错误，但有额外性能开销
- 当前状态：未改，待在稳定性确认后再清理

### 9. 小样本高指标是否能仅靠通信层达到

- 现状：
  - communication-only: `train 3.12%`, `test 1.56%`
  - full-finetune (aggressive): `train 100%`, `test 3.12%`
  - full-finetune (conservative): `train 12.50%`, `test 15.62%`
  - larger-sample full-finetune (`512/256`): `train 11.91%`, `test 13.28%`
  - full-data full-finetune (`7473/1319`): `train 34.60%`, `test 26.76%`
- 风险：
  - 当前 `64` 样本设置下，same-split 指标和 held-out 指标表现差异极大
  - 如果目标是 held-out `test > 80%`，当前没有证据表明该目标可在现有数据量与方法下实现
- 当前状态：工程链路已修复，指标结论已明确，后续需要按目标定义决定下一轮方向

### 10. 训练后评测已改为无 checkpoint live eval

- 文件：`src/cli/train.py`、`src/cli/evaluate.py`
- 新行为：
  - 训练结束后直接复用内存中的 `system` 做 eval
  - probe 配置显式关闭 final checkpoint 保存
  - 输出目录只保留轻量文件
- 当前状态：已修复并经多轮 probe 验证

## 本轮新增配置

- `configs/experiments/gsm8k_5agent_probe64_comm_only.yaml`
- `configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml`
- `configs/experiments/gsm8k_5agent_probe64_full_finetune_lowlr4.yaml`
- `configs/experiments/gsm8k_5agent_probe512_full_finetune_lowlr2.yaml`
- `configs/experiments/gsm8k_5agent_fulltrain_fulltest_fullft_lowlr1.yaml`

它们用于当前的小样本快速迭代，不替代全量训练配置。
