# Ours JSON Log Format

这份文档说明 `ours` 在评测阶段生成的两个主要 JSON 文件格式：

- `eval_results.json`
- `agent_logs.json`

这两个文件通常保存在同一个 checkpoint 目录下，例如：

```text
outputs/gsm8k_qwen3-8b_xxx/
```

---

## 1. `eval_results.json`

这个文件用于保存 `ours` 的主评测结果。它既包含整体指标，也包含逐条样本的结果。

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

- `method`
  当前方法名称。固定为 `ours_trained_multi_agent`。

- `task`
  当前评测任务名称，例如 `gsm8k`。

- `metrics`
  整体评测指标。

- `parameters`
  当前评测时使用的参数。

- `world_size`
  当前评测使用的进程数，也就是分布式评测时的总进程数。

- `samples`
  每条样本的结果列表。

### `metrics` 字段

```json
{
  "accuracy": 12.0,
  "correct": 12,
  "total": 100,
  "time_seconds": 345.2
}
```

字段说明：

- `accuracy`
  准确率，单位是百分比。

- `correct`
  答对的样本数。

- `total`
  总评测样本数。

- `time_seconds`
  当前评测总耗时，单位是秒。

### `parameters` 字段

当前会记录的典型字段包括：

```json
{
  "config_path": "outputs/.../config.yaml",
  "checkpoint_path": "outputs/.../final_model.pt",
  "max_samples": 100,
  "generation_max_new_tokens": 16384,
  "inference_mode": "chat_with_prefix",
  "use_terminal_prefix": true,
  "do_sample": false,
  "write_agent_logs": true,
  "worker": 2,
  "config": { ... }
}
```

字段说明：

- `config_path`
  当前使用的配置文件路径。

- `checkpoint_path`
  当前使用的 checkpoint 路径。

- `max_samples`
  本次评测最多使用多少条测试样本。`-1` 或 `null` 一般表示全量。

- `generation_max_new_tokens`
  当前推理时允许生成的最大 token 数。

- `inference_mode`
  当前终端 agent 的推理模式，例如 `chat_with_prefix`。

- `use_terminal_prefix`
  终端 agent 是否使用来自上游 agent 的 latent prefix。

- `do_sample`
  是否使用采样生成。

- `write_agent_logs`
  是否额外生成 `agent_logs.json`。

- `worker`
  当前评测使用的进程数。这里的 `worker` 本质上对应 `torchrun --nproc_per_node`。

- `config`
  当前完整配置文件内容，直接嵌入进结果 JSON，便于复现。

### `samples[*]` 字段

每条样本的结构如下：

```json
{
  "question": "...",
  "gold": "18",
  "prediction": "9",
  "generation": { ... },
  "correct": false,
  "agent_log": { ... }
}
```

字段说明：

- `question`
  原始问题文本。

- `gold`
  标准答案。

- `prediction`
  从模型生成结果中抽取出的最终答案。

- `generation`
  当前样本的完整生成记录。

- `correct`
  当前样本是否答对。

- `agent_log`
  当前样本对应的 agent 级结构化日志。这个字段与 `agent_logs.json` 中的对应样本信息基本一致。

### `samples[*].generation` 字段

当前结构示意如下：

```json
{
  "generated_text": "...",
  "finish_reason": "eos",
  "generated_token_count": 256,
  "stopped_early": true
}
```

字段说明：

- `generated_text`
  终端 agent 的完整生成文本。

- `finish_reason`
  生成停止原因。常见值包括：
  - `eos`
  - `max_new_tokens`

- `generated_token_count`
  本次实际生成了多少个 token。

- `stopped_early`
  是否在达到 `max_new_tokens` 之前提前结束。

---

## 2. `agent_logs.json`

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

字段说明：

- `method`
  当前方法名称。

- `task`
  当前任务名称。

- `parameters`
  当前评测参数。

- `samples`
  每条样本的 agent 级日志。

### `samples[*]` 字段

每条样本的结构如下：

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

这几个字段与 `eval_results.json` 的单样本记录含义一致，不再重复。

其中最重要的是 `agents` 字段。

---

## 3. `agents[*]` 字段

`agents` 是一个列表，表示当前样本中每个 agent 的结构化执行日志。

不同 agent 的字段略有不同，取决于它是非终端 agent 还是终端 agent。

### 3.1 非终端 agent

非终端 agent 一般会记录类似下面的字段：

```json
{
  "agent_id": 0,
  "role_name": "reader",
  "output_type": "latent",
  "system_prompt": "...",
  "received_upstream_prefix": false,
  "upstream_prefix": null,
  "hidden_trajectory": {
    "name": "hidden_trajectory",
    "shape": [1, 16, 1024],
    "norm": 123.4,
    "mean": 0.01,
    "std": 0.98
  },
  "compressed_prefix": {
    "name": "compressed_prefix",
    "shape": [1, 8, 1024],
    "norm": 45.6,
    "mean": 0.02,
    "std": 0.87
  }
}
```

字段说明：

- `agent_id`
  agent 在图中的编号。

- `role_name`
  当前 agent 的角色名称。

- `output_type`
  输出类型。非终端 agent 一般为 `latent`。

- `system_prompt`
  当前 agent 的 system prompt。

- `received_upstream_prefix`
  是否接收到了来自上游 agent 的 prefix。

- `upstream_prefix`
  上游 prefix 的统计摘要。如果没有上游输入，通常为 `null`。

- `hidden_trajectory`
  当前 agent 内部 latent trajectory 的统计摘要。

- `compressed_prefix`
  当前 agent 压缩后发送给下游的 latent prefix 统计摘要。

### 3.2 终端 agent

终端 agent 一般会记录类似下面的字段：

```json
{
  "agent_id": 4,
  "role_name": "solver",
  "output_type": "text",
  "system_prompt": "...",
  "received_upstream_prefix": true,
  "upstream_prefix": {
    "name": "upstream_prefix",
    "shape": [1, 8, 1024],
    "norm": 51.2,
    "mean": 0.01,
    "std": 0.93
  },
  "inference_mode": "chat_with_prefix",
  "used_upstream_prefix": true,
  "generated_text": "...",
  "generation": {
    "generated_text": "...",
    "finish_reason": "eos",
    "generated_token_count": 128,
    "stopped_early": true
  }
}
```

字段说明：

- `output_type`
  终端 agent 的输出类型一般为 `text`。

- `inference_mode`
  当前终端 agent 采用的推理模式。

- `used_upstream_prefix`
  终端 agent 在推理时是否真正使用了上游 prefix。

- `generated_text`
  当前终端 agent 的直接生成文本。

- `generation`
  终端 agent 的完整生成记录，包含完整文本和停止信息。

---

## 4. Tensor 摘要字段格式

像 `upstream_prefix`、`hidden_trajectory`、`compressed_prefix` 这类 tensor 摘要，统一采用如下结构：

```json
{
  "name": "compressed_prefix",
  "shape": [1, 8, 1024],
  "norm": 45.6,
  "mean": 0.02,
  "std": 0.87
}
```

字段说明：

- `name`
  当前张量摘要的名称。

- `shape`
  张量形状。

- `norm`
  张量的 L2 范数。

- `mean`
  张量元素均值。

- `std`
  张量元素标准差。

这些字段的作用是帮助分析 latent communication 的数值状态，而不是保存完整 tensor。

---

## 5. 当前注意事项

- `worker` 在 `ours eval` 中表示评测进程数，不是线程数。
- `generation.generated_text` 现在保存完整文本，不再依赖样本级顶层 `generated_text`。
- `agent_logs.json` 比 `eval_results.json` 更适合做调试分析，因为它保留了每个 agent 的中间状态摘要。
- 如果未来继续调整 JSON 结构，这份文档也应同步更新。
