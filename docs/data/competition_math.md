# competition_math

## Source

- Hugging Face: `qwedsacf/competition_math`
- 当前实现读取其唯一 `train` split

## Runtime Behavior

- 训练任务名：`competition_math`
- `question <- problem`
- `answer <- solution` 全文
- 额外保留：`level`、`type`

训练监督直接使用完整 `solution` 文本。评测与训练期 probe acc 则单独对模型生成文本做答案抽取，优先匹配 `\boxed{...}`，其次回退到常见的 `final answer is ...` 形式。

## Train-Time Probe

这个任务默认不做正式 `test` / `validation` eval。当前正式配置 `competition_math_5agent.yaml` 里 `training_probe.samples = 0`，因此不会默认切出 probe 子集。

如果改用 debug 配置，或手动把 `training_probe.samples` 设为正数，则会从 `train` split 中固定留出对应数量的样本作为 `training_probe`：

- `train_main`：参与梯度更新
- `probe`：不参与训练，只用于观察 acc 是否随 step 上升

probe 子集由 `training_probe.seed` 控制，并在启用时写入输出目录下的 `probe_split.json`。

## Outputs

当 `training_probe.write_predictions_json = true` 时，训练目录会额外产出：

- `probe_split.json`
- `probe_history.json`
