# Latent-MAS 训练流程说明

## 摘要

本文档描述当前仓库版本里真实可运行的训练 pipeline。它既说明方法层的核心对象，也说明工程层实际发生的步骤：配置如何进入 `train.py`，哪些模块参与训练，训练后如何直接做 live eval，以及输出目录里最终会留下什么产物。

当前版本的训练入口只有一个：[src/cli/train.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/train.py)。同一套入口通过 experiment config 切换不同训练策略，最常用的是：

- `communication_only`：只训练 `compressor + adjacency`
- `full_finetune`：在上述基础上同时训练 backbone

默认评测入口是 [src/cli/evaluate.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/evaluate.py)。训练脚本也可以在同一进程里直接复用内存中的 `system` 做 post-train live eval，而不必先保存 checkpoint。

## 1. Pipeline 总览

一次标准训练运行的主链路如下：

1. 读取 YAML 配置并做兼容归一化。
2. 初始化单卡或 DDP 环境。
3. 构建 [MultiAgentSystem](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/pipeline/multi_agent_system.py)。
4. 加载数据集，构造 `DataLoader`。
5. 对 question / answer 分别 tokenize。
6. 执行多 agent 前向，得到 task loss 与 graph loss。
7. 反向传播并更新可训练参数。
8. 按配置决定是否保存中间 checkpoint。
9. 若 `evaluation.run_after_train=true`，直接对内存中的系统做 train/test split 的 live eval。
10. 按配置决定是否保存 `final_model.pt`。

其中同一条 pipeline 支持：

- 小样本 probe：例如 `probe64`、`probe512`
- 全量训练：例如 `fulltrain/fulltest`
- 无 checkpoint 的轻量闭环：训练结束后直接输出 `eval_results*.json`

## 2. 配置如何驱动训练

配置加载逻辑在 [src/utils/config.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/utils/config.py)。

当前配置里最关键的段落有：

- `model`
- `graph`
- `compressor`
- `training`
- `reasoning`
- `evaluation`
- `output`
- `report`

### 2.1 `training` 相关字段

当前版本最关键的训练字段包括：

- `task`
- `train_strategy`
- `input_mode`
- `batch_size`
- `lr`
- `epochs`
- `max_seq_len`
- `lambda_add`
- `lambda_drop`
- `lambda_sparse`
- `gradient_accumulation_steps`
- `drop_last`
- `save_interval`
- `save_final_checkpoint`

其中：

- `train_strategy=communication_only` 时，只优化 `compressor + adjacency`
- `train_strategy=full_finetune` 时，还会把基础模型参数加入 optimizer
- `input_mode=chat_with_prefix` 时，终端 agent 的 teacher-forcing 输入和 eval prompt 都走 chat 模板路径

### 2.2 `evaluation` 相关字段

训练后 live eval 由 `evaluation` 段驱动，最关键的是：

- `run_after_train`
- `max_new_tokens`
- `batch_size`
- `train_probe_samples`
- `test_probe_samples`
- `inference_mode`
- `use_terminal_prefix`
- `do_sample`
- `write_agent_logs`

这使得同一份训练配置可以显式描述：

- 训练后要不要马上评测
- 评测跑 `train` / `test` 各多少样本
- 终端 agent 是否使用上游 prefix
- 是否额外生成 `agent_logs*.json`

## 3. 数据与批次

数据集逻辑在 [data/dataset.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/data/dataset.py)。

当前支持的任务包括：

- `gsm8k`
- `arc_easy`
- `arc_challenge`

训练 dataloader 的 `collate_fn` 很轻，只返回：

- `questions: list[str]`
- `answers: list[str]`

真正的 tokenization 发生在训练循环内部：

- question 先通过 `system.base_model.tokenize(...)`
- answer 再单独 tokenize
- 当 `training.input_mode == "chat_with_prefix"` 时，会额外给 answer 追加 EOS，保证终端 answer 监督与 chat prompt 对齐

这意味着当前训练不是把一个完整 prompt 串提前拼好再统一切片，而是：

1. question 单独编码
2. answer 单独编码
3. 到终端 agent 的 `forward_for_loss()` 时再按模式拼接

## 4. 模型组成与可训练参数

顶层系统在 [src/pipeline/multi_agent_system.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/pipeline/multi_agent_system.py)。

### 4.1 Base Model

[BaseModelWrapper](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/base_model.py) 负责：

- 加载 Hugging Face causal LM 与 tokenizer
- 提供普通前向
- 提供 latent reasoning 接口
- 提供 prefix embedding 注入能力

### 4.2 Agents

[Agent](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/agent.py) 自身不拥有独立 backbone 参数，而是共享同一个 base model。每个 agent 主要由角色 JSON 决定：

- `role_name`
- `system_prompt`
- `reasoning_steps`
- `compress_last_k`

### 4.3 Compressor 与 Adjacency

真正属于 communication layer 的可训练模块是：

- [LatentCompressor](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/compressor.py)
- [LearnableAdjacency](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/graph/adjacency.py)

### 4.4 当前哪些参数会被优化

`MultiAgentSystem.get_trainable_params()` 的真实行为是：

- `communication_only`：返回 `compressor.parameters() + adjacency.parameters()`
- `full_finetune`：额外包含 `base_model.model.parameters()`

因此，“当前版本只训练 communication layer”并不是普遍事实；它只对 `communication_only` 配置成立。

## 5. 单步训练前向如何展开

多 agent 执行逻辑在 [src/graph/dag_executor.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/graph/dag_executor.py)。

### 5.1 非终端 agent

对每个非终端节点，执行器会：

1. 根据 soft adjacency 聚合上游 prefix。
2. 调用 `agent.reason(...)` 做 latent reasoning。
3. 取最后 `k` 个 hidden states。
4. 送入 shared compressor，得到固定长度 prefix。
5. 把该 prefix 缓存给后续下游节点使用。

当前消息聚合使用的是加权求和：

$$
z_j = \sum_{i \in \mathcal{N}(j)} A_{ij} P_i
$$

不再做按边权和归一化。

### 5.2 终端 agent 训练分支

终端 agent 在训练时走 `forward_for_loss(...)`。

若 `input_mode=legacy_plain_with_prefix`，输入逻辑近似为：

$$
[z_T ; p_T ; x ; y]
$$

若 `input_mode=chat_with_prefix`，会先把 `system_prompt + question` 通过 chat template 变成 prompt，再把 `answer_ids` 追加在 prompt 之后。

模型返回 logits 后，`MultiAgentSystem.forward()` 会：

1. 根据 question/prompt 长度构造 labels
2. 计算 task CE loss
3. 计算 graph regularization loss
4. 相加得到总损失

## 6. 损失函数

总损失由两部分组成：

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \mathcal{L}_{graph}
$$

其中：

- `task_loss`：终端答案位置上的交叉熵
- `graph_loss`：由 `lambda_add`、`lambda_drop`、`lambda_sparse` 控制的图结构正则

图结构项并不替代任务监督；它的作用是约束 learned adjacency 不要无约束漂移。

## 7. DDP 与优化步骤

分布式逻辑也在 [src/cli/train.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/train.py)。

当前实现里：

- backbone 在 `full_finetune` 下用 DDP 包装
- compressor 用 DDP 包装
- adjacency 很小，不单独包 DDP，而是在 backward 后手动 `all_reduce` 它的梯度

每个更新步大致是：

1. 前向得到 `loss`
2. `loss / grad_accum_steps` 后 backward
3. DDP 场景下同步 adjacency 梯度
4. 计算 grad norm
5. `clip_grad_norm_`
6. `optimizer.step()`
7. 记录 wandb 和 `loss_log.csv`

## 8. 训练后 Live Eval

这是当前版本相对早期实现的最大变化之一。

当 `evaluation.run_after_train=true` 时，训练脚本不会强制先保存 checkpoint 再启动独立评测进程，而是直接调用 [evaluate_loaded_system](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/evaluate.py)：

- 复用内存中的 `system`
- 分别对 `train` / `test` split 做评测
- 写出 `eval_results_train.json` 与 `eval_results.json`
- 可选写出 `agent_logs_train.json` 与 `agent_logs.json`

这条路径特别适合：

- 小样本 probe
- 不想落超大 `.pt` 文件的全参实验
- 只保留轻量结果文件的快速迭代

## 9. 输出目录与产物

输出目录由 [src/utils/output_paths.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/utils/output_paths.py) 生成带时间戳的路径，例如：

```text
outputs/gsm8k_qwen3-8b_probe64_comm_only_YYYYMMDD_HHMMSS/
```

当前版本常见产物包括：

- `config.yaml`
- `loss_log.csv`
- `eval_results_train.json`
- `eval_results.json`
- `agent_logs_train.json` 或 `agent_logs.json`（如果启用）
- `checkpoint_step*.pt`（仅 `save_interval > 0` 时）
- `final_model.pt`（仅 `save_final_checkpoint=true` 时）

因此 `final_model.pt` 不是当前版本的必有产物。

## 10. 当前常用配置族

最近一轮实验里，最常见的配置分成三组：

- 基础 5-agent 配置：[configs/experiments/gsm8k_5agent.yaml](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/experiments/gsm8k_5agent.yaml)
- probe 配置：`probe64`、`probe512`
- full-data 配置：[configs/experiments/gsm8k_5agent_fulltrain_fulltest_fullft_lowlr1.yaml](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/experiments/gsm8k_5agent_fulltrain_fulltest_fullft_lowlr1.yaml)

这些配置共享同一个 train/eval 入口，只在训练策略、样本规模、是否保存 checkpoint、以及 post-train eval 参数上不同。

## 11. 相关文档

如果你在看当前版本的 train pipeline，建议一起看：

- [method.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/method.md)
- [agent_workflow.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/agent_workflow.md)
- [prompt_flow.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/prompt_flow.md)
- [ours_json_log_format.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/ours_json_log_format.md)
- [training_pipeline_risks.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/training_pipeline_risks.md)
- [docs/README.md](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/README.md)
