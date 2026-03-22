# HumanEval

## Dataset Source

- Hugging Face dataset: `openai_humaneval`
- underlying source split fetched by code: `test`
- local split policy in this repo:
  - `train` = first `60%`
  - `test` = remaining `40%`

代码注册位置见 [dataset.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/data/dataset.py) 中的 `TASK_CONFIGS["humaneval"]` 与 `_apply_humaneval_split(...)`。

这不是完整官方 HumanEval benchmark 划分，而是当前仓库里的本地 `debug_60_40` 运行方式。

## Installation

基础依赖：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

`human_eval` 官方 harness 额外需要单独准备。当前仓库实测更稳的方式不是直接把它装成 editable package，而是：

```bash
git clone https://github.com/openai/human-eval /tmp/human-eval
```

然后在运行 `evaluate.py` 前注入：

```bash
export PYTHONPATH=/tmp/human-eval${PYTHONPATH:+:$PYTHONPATH}
```

原因是官方仓库的打包元数据比较旧，在当前环境下直接 `uv pip install -e` 可能会因为 `pkg_resources` 或 console script 元数据报错。

## Train

当前实验配置：

```bash
configs/experiments/humaneval_5agent.yaml
```

但这份配置的原始参数在两卡上容易 OOM。当前实际跑通的一组安全参数是：

- `training.batch_size = 1`
- `training.max_seq_len = 1024`
- `evaluation.run_after_train = false`
- `report.use_wandb = false`

示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/tmp/human-eval${PYTHONPATH:+:$PYTHONPATH} \
uv run --python .venv/bin/python torchrun \
  --master_port=29611 \
  --nproc_per_node=2 \
  src/cli/train.py \
  --config /tmp/humaneval_5agent_train_safe.yaml
```

如果需要，可以先基于现有 YAML 生成一份临时安全配置再训练。

## Eval

`HumanEval` 评测复用 [evaluate.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/evaluate.py)，但它不会走字符串 exact-match，而是：

1. 生成 completion
2. 写出 `humaneval_samples.jsonl`
3. 调用官方 `human_eval` functional correctness harness
4. 汇总 `pass@k`

one-shot eval 示例：

```bash
CKPT_DIR=outputs/your_humaneval_run
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/tmp/human-eval${PYTHONPATH:+:$PYTHONPATH} \
uv run --python .venv/bin/python torchrun \
  --master_port=29612 \
  --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config "$CKPT_DIR/config.yaml" \
  --checkpoint "$CKPT_DIR/final_model.pt" \
  --split test \
  --max-new-tokens 512 \
  --inference-mode chat_with_prefix \
  --do-sample \
  --worker 2 \
  --batch-size 1
```

如果想做多样本 `pass@k`，需要在配置里增大：

- `evaluation.num_samples_per_task`
- `evaluation.pass_at_k`

但评测耗时会明显增加。

## Metric

- official `pass@k` from `human_eval`

## Output Artifacts

HumanEval 评测会产出：

- `eval_results.json`
- `agent_logs.json`
- `agent_log/<role>.json`
- `humaneval_samples.jsonl`
- `humaneval_problems.jsonl`
- `humaneval_samples.jsonl_results.jsonl`
