# 项目变更记录

本文档汇总了本次会话中从仓库初始状态到当前状态的主要修改，按主题整理，方便后续复现、交接和审阅。

## 1. Git 与分支管理

- 检查了主仓库的 git 状态、当前分支和本地提交身份。
- 在主仓库中创建并切换到 `toby` 分支。
- 为当前主仓库设置了本地 git 身份：
  - `user.name = TobyYang7`
  - `user.email = tobyyang7@outlook.com`
- 新建了一个本地 skill，用于后续在 git 提交前自动检查并修正当前仓库的本地 git 身份，避免误用全局配置。

## 2. 文档与论文资料

- 下载原始论文 PDF，并保存为 `docs/reference/paper.pdf`。
- 产出并整理了以下文档：
  - `docs/reference/paper_codebase_analysis.md`
  - `docs/training_pipeline.md`
  - `docs/training_pipeline_risks.md`
- 将训练流程文档改为中文表述，并补充了可渲染的数学公式。
- 新增本文档，用于统一记录本次改动历史。

## 3. 环境与依赖管理

- 使用 `uv` 为仓库创建并维护 `.venv` 环境。
- 新增 `.python-version`，固定 Python 版本。
- 更新 `requirements.txt`，补充训练、测试、PDF 解析、wandb 等依赖。
- 修复了 PDF 读取环境缺少依赖的问题，支持读取 `docs/reference/paper.pdf`。
- 统一约定通过 `HF_HOME` 使用默认 Hugging Face 缓存目录，而不是在配置里硬编码本地模型目录。

## 4. 训练配置与训练链路

- 将实验配置从 `Qwen/Qwen3-0.6B` 调整为 `Qwen/Qwen3-8B`。
- 移除了配置中对模型 `cache_dir` 的硬编码依赖，改为只使用模型名加载。
- 为训练脚本增加了最小样本数校验，避免 DDP 下出现 dataloader 长度为 0 的情况。
- 实际跑通了两卡训练，并完成了一次全量 GSM8K 训练实验：
  - 模型：`Qwen/Qwen3-8B`
  - 训练集：GSM8K 全量 `7473`
  - 测试集：GSM8K 全量 `1319`
- 训练阶段为了提高吞吐，将全量实验调整为：
  - 每卡 batch size 增大
  - 双卡 DDP
  - 先跑 1 个完整 epoch 做完整评估
- 当前已保存的训练产物包括：
  - `outputs/gsm8k_qwen3-8b_20260319_034138/final_model.pt`
  - `outputs/gsm8k_qwen3-8b_20260319_034138/loss_log.csv`

## 5. 模型与推理逻辑修改

- 修改 `BaseModelWrapper`：
  - 支持在不提供 `cache_dir` 时直接按模型名从默认 Hugging Face 缓存加载。
  - 增加离线回退逻辑，在网络受限时使用本地缓存。
- 修改 `Agent.generate_answer()`：
  - 为生成结果增加元信息。
  - 区分生成是因为 `eos` 提前结束，还是因为达到 `max_new_tokens` 被截断。
- 修改 `DAGExecutor` 和 `MultiAgentSystem`：
  - 将上述生成元信息从 agent 层向上透传，便于评估脚本记录。

## 6. 评测与 baseline

- 增加了两个显式 baseline 脚本：
  - 裸模型单模型 baseline
  - 原论文 `LatentMAS` baseline
- 对 baseline 输出格式进行了统一：
  - `method`
  - `metrics`
  - `parameters`
  - `samples` 或 `raw_result`
- 为 `single model baseline` 修复了以下问题：
  - `--max-samples -1` 现在表示全量样本
  - 批量生成时设置 `padding_side='left'`
  - 结果中记录生成停止原因和采样参数
- 当前已完成的全量 baseline 结果：
  - `outputs/baselines/single_model_qwen3-8b_all.json`
  - 全量 GSM8K 测试集 `1319` 条，正确 `187` 条，准确率约 `14.18%`

## 7. 多进程评测支持

- 将 `ours` 的评测脚本改造成支持 `torchrun` 多进程分片评测：
  - 每个 rank 只处理一部分样本
  - 最后由主进程聚合结果并保存 JSON
- 将 `single model baseline` 也改造成多进程分片评测，并完成了双进程 smoke test。
- 将 `paper LatentMAS baseline` 也加入了分片参数和聚合逻辑：
  - 修改了 `LatentMAS/run.py`
  - 修改了 wrapper 侧的聚合流程

## 8. wandb 上报

- 新增 `src/utils/reporting.py`，统一处理 wandb 初始化、日志记录和结束逻辑。
- 为训练脚本接入 wandb：
  - `src/cli/train.py`
  - `src/cli/multi_train.py`
- 在实验配置中加入 `report` 字段：
  - `use_wandb`
  - `project`
  - `key_env`
  - `env_file`
  - `mode`
- 实现了从仓库根目录 `.env` 中读取 `WANDB_API_KEY` 的逻辑。
- 当前行为：
  - 若环境变量或 `.env` 中存在 `WANDB_API_KEY`，则启用 wandb
  - 若不存在，则打印提示并安全跳过，不影响训练

## 9. 测试与验证

- 新增并更新了多组测试：
  - `tests/test_base_model.py`
  - `tests/test_baseline_scripts.py`
  - `tests/test_latentmas_cache_compat.py`
  - `tests/test_training_utils.py`
  - `tests/test_dag_executor.py`
- 这些测试覆盖了：
  - 默认 HF 缓存加载逻辑
  - baseline 输出路径和 JSON 解析
  - 论文实现对 `DynamicCache` 的兼容
  - DDP 最小样本数校验
  - generation metadata 的透传

## 10. 目录结构重构

- 将原本放在 `scripts/` 目录中的 Python 入口统一迁移到 `src/cli/`：
  - `train.py`
  - `multi_train.py`
  - `evaluate.py`
  - `run_baseline_single_model.py`
  - `run_baseline_paper_latentmas.py`
  - `run_pipeline.py`
  - `plot_loss.py`
- 将 `scripts/` 收敛为只放 shell 包装脚本：
  - `train.sh`
  - `multi_train.sh`
  - `evaluate.sh`
  - `eval_single.sh`
  - `eval_paper.sh`
  - `run_pipeline.sh`
  - `plot_loss.sh`
  - `launch_ddp.sh`
- 更新了 README 中的目录说明与入口说明。

## 11. 统一改为 uv run

- 所有 shell 入口统一改为使用：

```bash
uv run --python .venv/bin/python ...
```

- `src/cli/run_pipeline.py` 内部也改为统一通过 `uv run` 启动单进程和多进程流程。
- README 的 setup 和 quick start 说明同步更新为 `uv` 工作流。

## 12. LatentMAS 嵌套仓库说明

- `LatentMAS/` 是单独 clone 的上游论文实现仓库，不纳入当前主仓库 `toby` 分支的提交范围。
- 为兼容当前环境和实验流程，对其做过若干本地修改，包括：
  - 模型名支持 `Qwen/Qwen3-8B`
  - `DynamicCache` 兼容
  - baseline 分片评测支持
- 这些修改应保存在 `LatentMAS/` 自己的分支中，而不混入当前主仓库提交。

## 13. 当前主仓库关注点

当前主仓库的主要成果可以概括为：

- 完成了环境、训练、评测、baseline、wandb、目录结构和 shell 入口的一整套工程化整理。
- 支持使用 `uv`、双卡训练、双卡分片评测、统一 pipeline 脚本、以及更完整的结果 JSON。
- 已经将训练与推理中若干关键元信息显式记录下来，便于后续分析与复现。

## 14. Probe64 修复与实验闭环

- 为 `probe64` 增加了真正的“训练后直接评测”闭环，不再依赖 checkpoint：
  - `src/cli/evaluate.py` 新增 `evaluate_loaded_system()`
  - `src/cli/train.py` 可在训练结束后直接对内存中的模型做 train/test eval
- 对 `probe64` 配置显式加入：
  - `training.save_final_checkpoint: false`
  - `evaluation.run_after_train: true`
  - `evaluation.write_agent_logs: false`
- 修复了 DDP 下 terminal generation 的直接调用问题：
  - `src/models/agent.py` 现在会对 `generate()` 使用 unwrapped helper model
- 新增/扩展测试覆盖：
  - `tests/test_evaluate_streaming.py`
  - `tests/test_agent_generation.py`
  - `tests/test_config.py`
- 已完成 3 轮 `probe64` 实验，均不落 checkpoint，只保留轻量文件：
- 已完成 4 轮 probe 实验，均不落 checkpoint，只保留轻量文件：
- 已完成 5 轮实验，均不落 checkpoint，只保留轻量文件：
  - communication-only: `train 3.12%`, `test 1.56%`
  - full-finetune aggressive: `train 100%`, `test 3.12%`
  - full-finetune low-lr short-run: `train 12.50%`, `test 15.62%`
  - larger-sample full-finetune: `train 11.91%` on `512`, `test 13.28%` on `256`
  - full-data full-finetune: `train 34.60%` on `7473`, `test 26.76%` on `1319`
- 详细版本结果已单独记录到：
  - `docs/records/experiments/probe64_experiment_log_2026-03-19.md`
