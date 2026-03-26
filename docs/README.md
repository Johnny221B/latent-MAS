# Docs README

## Main Docs

这些文件描述当前代码实际行为，应该与运行时保持同步：

- `training_pipeline.md`
- `method.md`
- `agent_workflow.md`
- `prompt_flow.md`
- `ours_json_log_format.md`

## Dataset Runbooks

`docs/data/` 记录不同 dataset 的安装方式和 train/eval 运行方式：

- `docs/data/README.md`
- `docs/data/gsm8k.md`
- `docs/data/arc.md`
- `docs/data/humaneval.md`
- `docs/data/competition_math.md`
- `docs/data/am_deepseek_r1_distilled.md`

当前数据模块源码位于 `src/data/`，并按数据集拆分为独立模块，不再使用根目录 `data/` 包。

## Plans

`docs/plans/` 保存面向实现的设计稿和实施计划，不是当前运行行为的最终来源。

## Records

当前仓库里尚未单独建立 `docs/records/` 目录；历史变更记录、实验日志、问题追踪仍分散在 `docs/` 根目录下。后续若继续积累，应优先迁移到 `docs/records/`，避免它们和主文档混淆。

## Current HumanEval Status

当前仓库已支持 `humaneval` 的训练/eval 调试链路，但它使用的是本地 `debug_60_40` 划分：

- `train` = 官方 HumanEval 题集前 `60%`
- `test` = 官方 HumanEval 题集后 `40%`

因此：

- 该链路适合调试与本地比较
- `pass@k` 仍由官方 `human_eval` harness 计算
- 结果不能直接当作完整官方 HumanEval 基准分数引用

## Current competition_math Status

当前仓库已支持 `competition_math` 训练接入，但默认不做正式 post-train eval：

- 正式配置 [`competition_math_5agent.yaml`](./../configs/experiments/competition_math_5agent.yaml) 当前将 `training_probe.samples` 设为 `0`，因此不会额外留出 probe 子集
- 调试配置 [`competition_math_5agent_debug.yaml`](./../configs/experiments/competition_math_5agent_debug.yaml) 将 `training_probe.samples` 设为 `100`，用于训练期间的 probe 监控
- 若启用 probe，`probe` 不参与梯度更新，训练过程中按 `global_step` 记录 `probe accuracy`
- 调试配置默认禁用 W&B；正式配置才会上报线上 run

## Current AM DeepSeek R1 Distilled Status

当前仓库已支持 `am_deepseek_r1_distilled` 训练接入：

- 数据源为 `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`
- 运行时会合并 `am_0.5M` 与 `am_0.9M`
- 当前只支持 `train` split
- 训练监督目标是 assistant 的完整输出，保留 `<think>` 和 `<answer>` 标签
- 当前需要先运行 `scripts/prepare_am_deepseek_r1_distilled.py` 生成本地 `train.jsonl`，再启动训练
- 正式实验配置为 [`am_deepseek_r1_distilled_5agent.yaml`](./../configs/experiments/am_deepseek_r1_distilled_5agent.yaml)，当前模型名与输出目录对应 `Qwen/Qwen3-4B`
- 若需要训练期间监控，应使用已有 `training_probe` 机制，而不是依赖数据集自带评测切分
