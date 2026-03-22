# Ours JSON Log Format

这份文档说明 `ours` 在评测阶段生成的两个主要 JSON 文件格式：

- `eval_results.json`
- `agent_logs.json`
- `agent_log/<role>.json`

对于 `humaneval`，当前目录下还会额外出现：

- `humaneval_samples.jsonl`
- `humaneval_problems.jsonl`
- `humaneval_samples.jsonl_results.jsonl`

这两个文件通常保存在同一个 checkpoint 目录下，例如：

```text
outputs/gsm8k_qwen3-8b_xxx/
```

如果使用 `evaluate.py --question "..." --output-dir <dir>` 或 [`scripts/inference.sh`](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/scripts/inference.sh)，这些文件会写到指定的 `output_dir`，而不是 checkpoint 目录本身。

---

## 1. `eval_results.json`

这个文件用于保存 `ours` 的主评测结果。对于答案匹配任务，它既包含整体指标，也包含逐条样本的结果。对于 `humaneval`，它会改为保存 `pass@k`、样本文件路径以及每条 completion。

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

对 `humaneval`，顶层还会额外带一个 `artifacts` 字段，用来记录 `samples.jsonl`、`problems.jsonl` 与官方结果文件路径。

### `metrics` 字段

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

字段说明：

- `accuracy`
  准确率，单位是百分比。

- `correct`
  答对的样本数。

- `total`
  总评测样本数。

- `time_seconds`
  当前评测总耗时，单位是秒。

- `avg_sample_seconds`
  平均每条样本推理耗时，单位是秒。

- `avg_generated_tokens`
  平均每条样本实际生成的 token 数。

- `avg_tokens_per_second`
  用 `generated_token_count / sample_seconds` 汇总得到的整体生成吞吐。

对 `humaneval`，`metrics` 的核心字段会变成：

```json
{
  "pass@1": 0.25,
  "pass@10": 0.6,
  "time_seconds": 120.0,
  "num_tasks": 65,
  "num_samples_per_task": 20,
  "total_samples": 1300,
  "pass_at_k": [1, 10],
  "split_scheme": "debug_60_40"
}
```

这里的 `split_scheme = "debug_60_40"` 明确表示当前仓库把官方单一 HumanEval 题集切成了本地 `60/40` train/test 调试划分，因此该结果不能直接视为完整官方榜单分数。

### `parameters` 字段

当前会记录的典型字段包括：

```json
{
  "config_path": "outputs/.../config.yaml",
  "checkpoint_path": "outputs/.../final_model.pt",
  "split": "test",
  "max_samples": 100,
  "question": null,
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
  本次评测最多使用多少条样本。`-1` 或 `null` 一般表示全量。

- `split`
  当前评测使用的是 `train` 还是 `test` split。

- `generation_max_new_tokens`
  当前推理时允许生成的最大 token 数。

- `question`
  若为 `null`，表示本次是常规数据集评测；若为字符串，表示本次是单题手工推理，该字段保存原始输入问题文本。

- `inference_mode`
  当前终端 agent 的推理模式，例如 `chat_with_prefix`、`legacy_plain_with_prefix` 或 `chat_with_text`。

- `use_terminal_prefix`
  终端 agent 是否使用来自上游 agent 的 latent prefix。若 `inference_mode = chat_with_text`，这个字段一般只反映调用参数，实际推理不会用到 latent prefix。

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
  "question_id": "gsm8k-test-7",
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

- `question_id`
  当前样本在原始数据集中的稳定 id。若使用单题手工推理，则会自动生成 `manual-<sha1前缀>` 形式的稳定 id。当前仓库会把它贯穿到所有主要评测日志中，便于 role 级对齐分析。

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
  "question_id": "gsm8k-test-7",
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

在 `chat_with_text` 模式下，非终端 agent 会改成文本日志，典型结构类似：

```json
{
  "agent_id": 0,
  "role_name": "reader",
  "output_type": "text_message",
  "system_prompt": "...",
  "received_upstream_prefix": false,
  "upstream_prefix": null,
  "received_upstream_texts": true,
  "upstream_texts": ["..."],
  "generated_text": "...",
  "generation": {
    "generated_text": "...",
    "finish_reason": "eos",
    "generated_token_count": 64,
    "stopped_early": true,
    "inference_mode": "chat_with_text",
    "used_upstream_prefix": false
  }
}
```

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

- `received_upstream_texts`
  当前终端 agent 是否接收到了来自上游 agent 的文本消息。

- `upstream_texts`
  当前终端 agent 接收到的文本消息列表。在 `chat_with_text` 模式下，这个字段比 `upstream_prefix` 更关键。

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
- `agent_log/<role>.json` 适合做按角色横向分析，因为它把同一角色在所有 `question_id` 上的输入/输出聚在一起。

---

## 5. `agent_log/<role>.json`

这是当前评测额外生成的一组按角色拆分的日志文件。目录结构类似：

```text
outputs/.../agent_log/
  reader.json
  planner.json
  solver.json
```

单个角色文件的结构如下：

```json
{
  "role_name": "reader",
  "samples": {
    "gsm8k-test-7": {
      "question_id": "gsm8k-test-7",
      "question": "...",
      "input": {
        "system_prompt": "...",
        "received_upstream_prefix": false,
        "upstream_prefix": null,
        "received_upstream_texts": false,
        "upstream_texts": []
      },
      "output": {
        "output_type": "text_message",
        "generated_text": "...",
        "generation": { ... },
        "hidden_trajectory": null,
        "compressed_prefix": null
      }
    }
  }
}
```

其中 `samples` 的 key 就是原始数据集里的 `question_id`。
- 如果未来继续调整 JSON 结构，这份文档也应同步更新。
