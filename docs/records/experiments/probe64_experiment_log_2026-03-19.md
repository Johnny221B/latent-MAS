# Probe64 Experiment Log

这份文档记录 2026-03-19 这轮 `probe64` 训练与推理修复后的实际实验结果。只保留轻量产物：`config.yaml`、`loss_log.csv`、`eval_results*.json`。

## Version 1: Communication-Only, 64 Epochs

- 配置：`configs/experiments/gsm8k_5agent_probe64_comm_only.yaml`
- 命令：`CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_probe64_comm_only.yaml --max_samples 64`
- 输出目录：`outputs/gsm8k_qwen3-8b_probe64_comm_only_20260319_152033`
- 轻量产物：
  - `config.yaml`
  - `loss_log.csv`
  - `eval_results_train.json`
  - `eval_results.json`

### 观察

- 训练工程链路正常：
  - 双卡 DDP 正常启动
  - 每卡 `batch_size=32`
  - 有效 batch size `64`
  - 无 checkpoint 文件落盘
  - 训练后 live eval 正常执行
- 资源占用：
  - 模型加载后显存约 `16.0 GB`
  - 训练峰值显存约 `61.8 GB`
- 损失：
  - `task_loss` 从约 `18.0` 下降到约 `1.375`
  - `adjacency` 仅有极小变化，主图结构基本保持先验

### 结果

- Train 64:
  - Accuracy: `3.12%` (`2/64`)
  - Avg sample seconds: `0.13`
  - Avg generated tokens: `64.0`
  - Tokens/sec: `493.75`
- Test 64:
  - Accuracy: `1.56%` (`1/64`)
  - Avg sample seconds: `0.12`
  - Avg generated tokens: `64.0`
  - Tokens/sec: `529.26`

### 结论

- communication-only 在当前实现下能优化训练损失，但完全达不到 probe accuracy 目标。
- 这一轮暴露出 `max_new_tokens=64` 太低，所有样本都被截在上限，accuracy 被进一步压低。

## Version 2: Full-Finetune, 1e-4, 64 Epochs

- 配置：`configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml`
- 命令：`CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml --max_samples 64`
- 输出目录：`outputs/gsm8k_qwen3-8b_probe64_full_finetune_20260319_152513`

### 观察

- 工程链路正常：
  - full-finetune + DDP 训练可运行
  - terminal generation 在 DDP 下可运行
  - 无 checkpoint 文件落盘
- 资源占用：
  - 训练峰值显存约 `125.6 GB`
- 损失：
  - `task_loss` 在约第 `24` 轮后接近 `0`
  - 明显记忆化训练集

### 结果

- Train 64:
  - Accuracy: `100.00%` (`64/64`)
  - Avg sample seconds: `0.08`
  - Avg generated tokens: `6.0`
  - Tokens/sec: `77.75`
- Test 64:
  - Accuracy: `3.12%` (`2/64`)
  - Avg sample seconds: `0.08`
  - Avg generated tokens: `4.5`
  - Tokens/sec: `58.41`

### 结论

- 这轮满足“同一批 64 个训练样本上的 inference 正确率 > 80%”。
- 这轮完全不满足 held-out generalization。
- 失败模式是严重过拟合：test 上经常直接输出训练集中记住的短答案，例如 `48`、`5`、`10`。

## Version 3: Full-Finetune, 1e-5, 4 Epochs

- 配置：`configs/experiments/gsm8k_5agent_probe64_full_finetune_lowlr4.yaml`
- 命令：`CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_probe64_full_finetune_lowlr4.yaml --max_samples 64`
- 输出目录：`outputs/gsm8k_qwen3-8b_probe64_full_finetune_lowlr4_20260319_153019`

### 观察

- 把 eval token cap 提到 `128` 后，不再出现 communication-only 那种统一卡死在 `64 token` 的截断模式。
- 低学习率、短训练避免了 Version 2 那种极端记忆化，但训练样本也没有被充分拟合。

### 结果

- Train 64:
  - Accuracy: `12.50%` (`8/64`)
  - Avg sample seconds: `0.17`
  - Avg generated tokens: `128.0`
  - Tokens/sec: `739.24`
- Test 64:
  - Accuracy: `15.62%` (`10/64`)
  - Avg sample seconds: `0.18`
  - Avg generated tokens: `128.0`
  - Tokens/sec: `715.14`

### 结论

- 这轮 held-out test 明显好于 Version 2，但仍远低于 `80%`。
- 在当前代码、模型和 `64` 样本设置下，没有证据表明 held-out `test 64 > 80%` 是可达的。

## Version 4: Larger Sample Follow-Up, 512 Train / 256 Test

- 配置：`configs/experiments/gsm8k_5agent_probe512_full_finetune_lowlr2.yaml`
- 命令：`CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_probe512_full_finetune_lowlr2.yaml --max_samples 512`
- 输出目录：`outputs/gsm8k_qwen3-8b_probe512_full_finetune_lowlr2_20260319_153823`

### 观察

- 工程链路保持稳定：
  - 双卡 DDP 正常
  - 无 checkpoint 文件落盘
  - 训练后 live eval 正常执行
- 资源占用：
  - 训练峰值显存约 `128.5 GB`
- 训练动态：
  - `512` train samples
  - 每卡 `batch_size=32`
  - 每轮 `8` 步，共 `2` 轮
  - 训练 loss 从 `17.0` 下降到 epoch 2 平均约 `0.93`

### 结果

- Train 512:
  - Accuracy: `11.91%` (`61/512`)
  - Avg sample seconds: `0.15`
  - Avg generated tokens: `97.06`
  - Tokens/sec: `640.36`
- Test 256:
  - Accuracy: `13.28%` (`34/256`)
  - Avg sample seconds: `0.15`
  - Avg generated tokens: `98.12`
  - Tokens/sec: `635.20`

### 结论

- 相比 `64` 样本 aggressive full-ft，这轮泛化更稳定，不再出现“train 100 / test 3”的极端记忆化。
- 但 same-split 和 held-out 指标都仍然很低，没有接近高阈值目标。

## Version 5: Full Train / Full Test, 7473 Train / 1319 Test

- 配置：`configs/experiments/gsm8k_5agent_fulltrain_fulltest_fullft_lowlr1.yaml`
- 命令：`CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_fulltrain_fulltest_fullft_lowlr1.yaml`
- 输出目录：`outputs/gsm8k_qwen3-8b_fulltrain_fulltest_fullft_lowlr1_20260319_155356`

### 观察

- 训练链路保持稳定：
  - 双卡 DDP 正常
  - 无 checkpoint 文件落盘
  - 训练后 live eval 正常执行
- 资源占用：
  - 模型加载后显存约 `16.0 GB`
  - 训练峰值显存约 `148.1 GB`
- 训练动态：
  - 全量 `7473` train samples
  - 每卡 `batch_size=32`
  - 每轮 `117` 步，单 epoch 约 `5m47s`
  - epoch 平均 loss 约 `0.8631`

### 结果

- Train 7473:
  - Accuracy: `34.60%` (`2586/7473`)
  - Avg sample seconds: `0.08`
  - Avg generated tokens: `7.12`
  - Tokens/sec: `89.57`
- Test 1319:
  - Accuracy: `26.76%` (`353/1319`)
  - Avg sample seconds: `0.08`
  - Avg generated tokens: `6.95`
  - Tokens/sec: `86.05`

### 结论

- 这是目前最稳定、也最有代表性的结果。
- 相比小样本 probe，full-data 训练显著提升了 held-out test 表现，但仍然没有接近 `80%`。

## Overall Summary

- 工程修复已经完成并验证：
  - 消息聚合改为直接加权和
  - 文档已收紧为只训练 communication layer
  - DDP 下的 helper access 与 `generate()` 路径已修复
  - 训练后 live eval 已支持，且不落 checkpoint 文件
- 当前实验性结论：
  - 如果指标定义为“训练后在同一批 64 个训练样本上做 inference”，Version 2 已达标
  - 如果指标定义为“held-out 64 test 样本也要 > 80%”，当前没有达标
  - 增加到 `512` train samples 后，held-out test 仍只有 `13.28%`
  - 增加到全量 `7473/1319` 后，held-out test 提升到 `26.76%`

## Candidate Next Steps

- 若目标是 same-split probe：Version 2 已足够证明训练/推理链路可用。
- 若目标是 held-out generalization：
  - 不应继续沿着 `64` 样本全参 SFT 硬调 epoch 与 lr
  - 更合理的方向是：
    - 保留基础模型能力，只训练更轻量的适配层
    - 或显式引入验证集早停
    - 或使用更多训练样本
