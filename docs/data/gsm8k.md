# GSM8K

## Dataset Source

- Hugging Face dataset: `openai/gsm8k`
- subset: `main`

代码注册位置见 [gsm8k.py](../../src/data/gsm8k.py) 与 [factory.py](../../src/data/factory.py)。

## Installation

只需要仓库已有依赖：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

`gsm8k` 本身通过 `datasets.load_dataset(...)` 在线下载，不需要额外单独安装脚本。

## Train

默认配置：

```bash
configs/experiments/gsm8k_5agent.yaml
```

示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run --python .venv/bin/python torchrun \
  --master_port=29611 \
  --nproc_per_node=2 \
  src/cli/train.py \
  --config configs/experiments/gsm8k_5agent.yaml
```

## Eval

示例：

```bash
CKPT_DIR=outputs/your_gsm8k_run
CUDA_VISIBLE_DEVICES=0,1 \
uv run --python .venv/bin/python torchrun \
  --master_port=29612 \
  --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config "$CKPT_DIR/config.yaml" \
  --checkpoint "$CKPT_DIR/final_model.pt" \
  --split test \
  --max-new-tokens 16384 \
  --inference-mode chat_with_prefix \
  --worker 2 \
  --batch-size 1
```

## Metric

- exact-match on extracted final numeric answer
