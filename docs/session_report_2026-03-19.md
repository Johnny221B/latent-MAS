# Session Report 2026-03-19

## 1. 本次主要修改

### 1.1 训练与评测链路

- 在 [src/cli/train.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/train.py) 中增加了训练后直接评测的闭环。
- 在 [src/cli/evaluate.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/evaluate.py) 中增加了 `evaluate_loaded_system()`，允许直接对内存中的模型做评测。
- probe/full-data 配置统一支持：
  - `evaluation.run_after_train: true`
  - `training.save_final_checkpoint: false`
  - 不再保存 `.pt` checkpoint，只保留轻量结果文件。

### 1.2 推理与 DDP 兼容

- 在 [src/models/agent.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/agent.py) 中修复了 DDP 下 terminal generation 直接调用 `generate()` 的兼容问题。
- 在 [src/models/base_model.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/base_model.py) 中补强了 helper model 访问逻辑，避免 DDP 包装后 helper 读取失效。
- 在 [src/models/compressor.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/compressor.py) 中修复了 mixed dtype 输入到 compressor 时的兼容问题。

### 1.3 方法实现与文档一致性

- 在 [src/communication/aggregator.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/communication/aggregator.py) 中去掉了原来的按入边权重和归一化，改为直接加权和。
- 在 [docs/method.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/method.md) 中把训练目标收紧为“当前默认只训练 communication layer”，避免继续误写成联合训练所有 agent reasoning。

### 1.4 配置、测试、文档

- 新增了多组 probe/full-data 配置：
  - `probe64`
  - `probe512`
  - `fulltrain/fulltest`
- 新增并更新了测试：
  - [tests/test_agent_generation.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/tests/test_agent_generation.py)
  - [tests/test_config.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/tests/test_config.py)
  - [tests/test_evaluate_streaming.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/tests/test_evaluate_streaming.py)
- 更新了版本记录与实验记录：
  - [docs/change_log_2026-03-19.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/change_log_2026-03-19.md)
  - [docs/current_version_issues_2026-03-19.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/current_version_issues_2026-03-19.md)
  - [docs/probe64_experiment_log_2026-03-19.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/probe64_experiment_log_2026-03-19.md)

## 2. 实验结果

### 2.1 小样本实验

- `probe64 communication-only`
  - train `3.12%`
  - test `1.56%`
- `probe64 full-finetune aggressive`
  - train `100.00%`
  - test `3.12%`
  - 结论：严重过拟合
- `probe64 full-finetune low-lr short-run`
  - train `12.50%`
  - test `15.62%`
- `probe512 full-finetune`
  - train `11.91%`
  - test `13.28%`

### 2.2 全量实验

- 配置：`configs/experiments/gsm8k_5agent_fulltrain_fulltest_fullft_lowlr1.yaml`
- 输出目录：`outputs/gsm8k_qwen3-8b_fulltrain_fulltest_fullft_lowlr1_20260319_155356`
- 全量 train `7473`
  - `2586/7473 = 34.60%`
- 全量 test `1319`
  - `353/1319 = 26.76%`
- 训练峰值显存约 `148.1GB`
- 没有保存任何 checkpoint 文件

## 3. 当前结论

- 工程层面的问题已经基本修通：
  - 训练可跑
  - 推理可跑
  - DDP 可跑
  - 训练后 live eval 可跑
  - 不会再存大 checkpoint
- 方法层面的结论也比较明确：
  - communication-only 在当前实现下效果很弱
  - 小样本 full-finetune 很容易记忆化
  - 增加样本后，held-out test 确实提升了，但当前最好结果也只有 `26.76%`

## 4. 当前还存在的问题

- 图结构学习几乎没有明显变化，adjacency 基本停留在先验附近，说明当前训练主要还是在学 backbone 或 compressor，而不是在学新的通信结构。
- full-finetune 的资源消耗非常高，双卡下峰值显存已经到 `148GB` 量级。
- 全量训练后的 test accuracy 仍远低于高目标，说明仅靠现在这套训练方式还不够。
- `find_unused_parameters=True` 仍然保留，会带来额外 DDP 开销。

## 5. 建议的下一步

- 如果目标是继续提升 test accuracy：
  - 优先考虑更轻量的适配方式，而不是继续全参硬训
  - 或者重新设计 communication mechanism，而不是让 adjacency 基本不动
  - 或者引入验证集早停与更系统的超参搜索
- 如果目标是论文/方法一致性：
  - 当前文档口径已经比之前更准确，但算法层仍需要决定到底要不要继续保留“learnable graph”作为核心卖点
