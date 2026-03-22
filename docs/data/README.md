# Dataset Runbook

这里汇总当前仓库不同 dataset 的来源、依赖和 train/eval 运行方式。

当前已记录：

- `gsm8k`
- `arc_easy`
- `arc_challenge`
- `humaneval`

## Files

- `gsm8k.md`
- `arc.md`
- `humaneval.md`

## Common Entry Points

训练入口：

```bash
src/cli/train.py
```

评测入口：

```bash
src/cli/evaluate.py
```

dataset 注册位置：

```bash
src/data/
```

当前已提供的现成实验 YAML：

- `configs/experiments/arc_easy_5agent.yaml`
- `configs/experiments/arc_challenge_5agent.yaml`
- `configs/experiments/humaneval_5agent.yaml`
- `configs/experiments/arc_easy_5agent_qwen3_4b_smoke.yaml`
- `configs/experiments/humaneval_5agent_qwen3_4b_smoke.yaml`
