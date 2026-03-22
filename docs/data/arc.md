# ARC

## Dataset Source

- Hugging Face dataset: `allenai/ai2_arc`
- supported subsets:
  - `ARC-Easy` -> `arc_easy`
  - `ARC-Challenge` -> `arc_challenge`

代码注册位置见 [arc.py](../../src/data/arc.py) 与 [factory.py](../../src/data/factory.py) 中的：

- `build_task_configs()["arc_easy"]`
- `build_task_configs()["arc_challenge"]`
- `_format_arc_question(...)`

## Installation

只需要仓库已有依赖：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

ARC 数据通过 `datasets.load_dataset(...)` 在线下载，不需要额外单独安装。

## Train

当前仓库已经提供两份 ARC 实验 YAML：

- [arc_easy_5agent.yaml](../../configs/experiments/arc_easy_5agent.yaml)
- [arc_challenge_5agent.yaml](../../configs/experiments/arc_challenge_5agent.yaml)
- [arc_easy_5agent_qwen3_4b_smoke.yaml](../../configs/experiments/arc_easy_5agent_qwen3_4b_smoke.yaml)

示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run --python .venv/bin/python torchrun \
  --master_port=29611 \
  --nproc_per_node=2 \
  src/cli/train.py \
  --config configs/experiments/arc_easy_5agent.yaml
```

这两份配置默认都会在训练结束后自动跑 `test` split 评测。

其中 `arc_easy_5agent_qwen3_4b_smoke.yaml` 是最小链路验证配置，默认使用 `Qwen/Qwen3-4B`、`batch_size=1`、较短 reasoning steps，并关闭 wandb。

## Eval

同样复用 [evaluate.py](../../src/cli/evaluate.py)。

示例：

```bash
CKPT_DIR=outputs/your_arc_run
CUDA_VISIBLE_DEVICES=0,1 \
uv run --python .venv/bin/python torchrun \
  --master_port=29612 \
  --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config "$CKPT_DIR/config.yaml" \
  --checkpoint "$CKPT_DIR/final_model.pt" \
  --split test \
  --max-new-tokens 64 \
  --inference-mode chat_with_prefix \
  --worker 2 \
  --batch-size 1
```

## Prompt Shape

ARC 不会把原始 `question` 裸传给模型。dataset 层会先把题面整理成：

```text
<question>

Choices:
A. ...
B. ...
C. ...
D. ...
```

训练和评测都会复用这一份格式化后的 `question` 字符串，因此二者看到的选项文本完全一致。

## Metric

- exact-match on extracted multiple-choice answer key
