# competition_math

## Source

- Hugging Face: `qwedsacf/competition_math`
- 当前实现读取其唯一 `train` split

## Runtime Behavior

- 训练任务名：`competition_math`
- `question <- problem`
- `answer <- solution` 中抽出的最终答案
- 额外保留：`level`、`type`

答案抽取优先匹配 `\boxed{...}`，其次回退到常见的 `final answer is ...` 形式。训练监督与训练期 probe acc 复用同一套规范化逻辑。

## Train-Time Probe

这个任务默认不做正式 `test` / `validation` eval，而是在训练入口里从 `train` split 固定留出 `100` 条样本作为 `training_probe`：

- `train_main`：参与梯度更新
- `probe100`：不参与训练，只用于观察 acc 是否随 step 上升

probe 子集由 `training_probe.seed` 控制，并写入输出目录下的 `probe_split.json`。

## Outputs

当 `training_probe.write_predictions_json = true` 时，训练目录会额外产出：

- `probe_split.json`
- `probe_history.json`
