# Ours JSON Log Format

这份文档说明 `ours` 在评测阶段生成的两个主要 JSON 文件格式：

- `eval_results*.json`
- `agent_logs*.json`

另外，当前版本还会把关键终端进度同步写到本地文本日志：

- `eval_progress.log`
- `train_progress.log`（如果是从训练脚本触发的 live eval）

它既适用于独立 checkpoint eval，也适用于训练结束后直接复用内存中模型做的 live eval。

## 1. 输出文件名

按 split 区分时，当前常见文件名是：

- `eval_results.json`：`test` split
- `eval_results_train.json`：`train` split
- `agent_logs.json`：`test` split 且 `write_agent_logs=true`
- `agent_logs_train.json`：`train` split 且 `write_agent_logs=true`
- `eval_progress.log`：评测过程中的关键终端进度副本
- `train_progress.log`：训练过程中的关键终端进度副本

这些文件通常保存在同一个 output 目录下，例如：

```text
outputs/gsm8k_qwen3-8b_probe64_comm_only_YYYYMMDD_HHMMSS/
```

## 2. `eval_results*.json`

这个文件用于保存主评测结果。它同时包含整体指标和逐样本结果。

### 顶层字段

```json
{
  "method": "ours_trained_multi_agent",
  "task": "gsm8k",
  "metrics": { ... },
  "parameters": { ... },
  "world_size": 2,
  "samples": [ ... ]
}
```

字段说明：

- `method`：固定为 `ours_trained_multi_agent`
- `task`：任务名，例如 `gsm8k`
- `metrics`：整体指标
- `parameters`：当前评测参数与完整 config 快照
- `world_size`：评测时使用的总进程数
- `samples`：逐条样本结果

### `metrics`

```json
{
  "accuracy": 12.0,
  "correct": 12,
  "total": 100,
  "time_seconds": 345.2,
  "avg_sample_seconds": 3.45,
  "avg_generated_tokens": 128.0,
  "avg_tokens_per_second": 37.1
}
```

其中：

- `accuracy` 是百分比
- `avg_generated_tokens` 是每条样本生成 token 数的平均值
- `avg_tokens_per_second` 是基于 `generated_token_count / sample_seconds` 的整体吞吐统计

### `parameters`

当前典型字段包括：

```json
{
  "config_path": "outputs/.../config.yaml",
  "checkpoint_path": null,
  "split": "train",
  "max_samples": 64,
  "generation_max_new_tokens": 128,
  "inference_mode": "chat_with_prefix",
  "use_terminal_prefix": true,
  "communication_mode": "latent_prefix",
  "text_message_edge_threshold": 0.5,
  "text_message_max_new_tokens": 512,
  "do_sample": false,
  "write_agent_logs": false,
  "worker": null,
  "batch_size": 32,
  "preview_limit": 0,
  "config": { ... }
}
```

说明：

- `checkpoint_path` 在 live eval 下可以是 `null`
- `split` 明确区分 train/test 指标
- `preview_limit=0` 表示跳过 preflight sample preview，直接进入 batch eval
- `config` 会嵌入完整 experiment config，方便复现

### `samples[*]`

每条样本的主结构如下：

```json
{
  "question": "...",
  "gold": "18",
  "prediction": "9",
  "generation": { ... },
  "correct": false
}
```

如果 `write_agent_logs=true`，同一条 sample 还会携带 `agent_log` 字段。

### `samples[*].generation`

当前 generation 字段通常包含：

```json
{
  "generated_text": "...",
  "finish_reason": "eos",
  "generated_token_count": 256,
  "stopped_early": true,
  "inference_mode": "chat_with_prefix",
  "used_upstream_prefix": true
}
```

其中：

- `finish_reason` 常见值是 `eos` 或 `max_new_tokens`
- `used_upstream_prefix` 记录终端 agent 是否真的消费了 terminal prefix
- 如果 `communication_mode=text_messages`，这里通常会是 `false`，因为终端 agent 不再读取 latent prefix

## 3. `agent_logs*.json`

这个文件用于保存更细粒度的 agent 级日志，主要用于调试和分析多智能体通信过程。

### 顶层字段

```json
{
  "method": "ours_trained_multi_agent",
  "task": "gsm8k",
  "parameters": { ... },
  "samples": [ ... ]
}
```

这里的 `parameters` 与主结果 JSON 保持一致，便于交叉对照。

### `samples[*]`

每条样本通常包含：

```json
{
  "question": "...",
  "gold": "18",
  "prediction": "9",
  "generation": { ... },
  "correct": false,
  "agents": [ ... ]
}
```

### `agents[*]`

非终端 agent 通常会记录：

- `agent_id`
- `role_name`
- `output_type = latent`
- `system_prompt`
- `received_upstream_prefix`
- `upstream_prefix`
- `hidden_trajectory`
- `compressed_prefix`

在 `communication_mode=text_messages` 的 eval 对照模式下，非终端 agent 会改成记录：

- `agent_id`
- `role_name`
- `output_type = text_message`
- `system_prompt`
- `upstream_text_messages`
- `generated_text`
- `generation`

终端 agent 通常会记录：

- `agent_id`
- `role_name`
- `output_type = text`
- `system_prompt`
- `received_upstream_prefix`
- `inference_mode`
- `used_upstream_prefix`
- `generated_text`
- `generation`

其中张量类字段不会保存完整数值，而是保存 shape / norm / mean / std 这类摘要统计。
而 `upstream_text_messages` 会保留结构化文本条目，典型字段是 `agent_id`、`role_name`、`content`、`edge_weight`。

## 4. 读取这些文件时要注意什么

- `eval_results_train.json` 和 `eval_results.json` 不能混为一个指标；它们代表不同 split。
- 如果 `train_probe_samples=0`，训练后可以只生成 `eval_results.json` / `agent_logs.json`，而没有 train split 对应文件。
- `checkpoint_path=null` 不表示结果无效，只表示它来自 live eval。
- `agent_logs*.json` 是可选产物，不是每次评测都会生成。
- `eval_progress.log` / `train_progress.log` 不是结构化 JSON；它们是给人追踪后台进度用的文本副本。
- 小样本 probe 时应明确区分 same-split 指标和 held-out 指标。
