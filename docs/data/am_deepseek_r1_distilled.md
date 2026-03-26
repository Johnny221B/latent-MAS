# am_deepseek_r1_distilled

## Source

- Hugging Face: `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`
- 当前实现会同时读取：
  - `am_0.5M`
  - `am_0.9M`
- 两个 subset 会先被预处理脚本下载并拼接，再写成本地 `train.jsonl`

## Prepare First

当前训练不会在 dataloader 内部直接联网下载 Hugging Face 数据。正确流程是先生成本地文件：

```bash
.venv/bin/python scripts/prepare_am_deepseek_r1_distilled.py
```

默认输出路径：

```text
data/am_deepseek_r1_distilled/train.jsonl
```

当前预处理脚本会直接下载仓库里的原始 `am_0.5M.jsonl` 与 `am_0.9M.jsonl` 文件，再逐行规范化写成本地训练文件，不再通过 `datasets.load_dataset(dataset_name, subset, ...)` 在线建表。

如果该文件不存在，训练时会直接报错并提示先运行预处理脚本。

## Runtime Behavior

- 训练任务名：`am_deepseek_r1_distilled`
- 支持 split：仅 `train`
- dataloader 直接读取本地 `data/am_deepseek_r1_distilled/train.jsonl`
- dataloader 走仓库内置的轻量 JSONL 读取器，不再调用 `datasets.load_dataset("json", ...)`
- 因此运行时不应再生成 `data/am_deepseek_r1_distilled/.hf_cache`
- 正常启动时也不应再看到 Hugging Face `Generating train split: ...` 这类建表日志
- `question <- messages` 中 `role == "user"` 的 `content`
- `answer <- messages` 中 `role == "assistant"` 的完整 `content`
- 当前本地训练文件只保留训练真正需要的稳定字段：`question_id`、`question`、`answer`、`subset`

当前读取器会先为 `train.jsonl` 建立行偏移索引，然后按需 seek 到单条样本读取；`--max_samples` 这类裁剪仍通过 dataset `select()` 生效，但不会触发额外缓存目录。

这里的监督目标不是抽取后的最终答案，而是完整 assistant 输出。当前实现会原样保留类似：

```text
<think>...</think><answer>...</answer>
```

这意味着训练时 teacher forcing 看到的是完整的思维与作答标签文本。

## Question IDs

当前 `question_id` 不直接使用题面原文，而是基于：

- `subset`
- `user content`
- `assistant content`

生成稳定哈希，格式为：

```text
am-r1-<12 hex chars>
```

这样可以避免超长 ID，并在两个 subset 合并后保持稳定可追踪。

## Train-Time Probe

该任务当前不提供正式 `test` / `validation` split。若需要训练期间的走势观测，应继续使用仓库已有的 `training_probe` 机制从 `train` 中留出 probe-only 样本，而不是依赖数据集自带评测切分。
