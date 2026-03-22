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
