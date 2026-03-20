# Latent-MAS 训练流程潜在问题分析

本文档只记录当前版本仍然有效、且会影响实验判断或工程扩展的风险。已经修复的历史问题应放到 `docs/records/`，不再留在这里冒充当前风险。

## 1. 图结构学习信号仍然偏弱

当前消息聚合已经改成加权求和，不再有旧版归一化的直接缺陷；但从最近几轮实验看，adjacency 仍然经常停留在先验附近，说明：

- 图结构学习信号依然偏弱
- 很多性能变化更可能来自 backbone 或 compressor
- “learnable graph” 是否真的学到任务相关协作，仍需谨慎解读

这属于方法风险，不是简单代码 bug。

## 2. `communication_only` 与 `full_finetune` 的结论不能混写

当前版本支持两种训练策略：

- `communication_only`
- `full_finetune`

如果不显式注明配置，实验结论很容易被误读。例如：

- `communication_only` 的差结果，不能直接证明 latent communication 无效
- `full_finetune` 的提升，也不能自动归因于图结构学习成功

因此所有实验汇总都必须同时写清：

- train strategy
- 样本规模
- train/test split
- 是否使用 terminal prefix

## 3. 小样本 probe 极易把 memorization 和 generalization 混为一谈

最近的 `probe64` 结果已经表明：

- same-split accuracy 可能很高
- held-out test accuracy 可能仍然很差

因此在小样本场景下，`train` 与 `test` 指标必须分开记录。只看一个数字会直接导致错误结论。

## 4. `chat_with_prefix` 是主路径，旧 plain 路径只能做诊断

当前目标评测路径应以 `chat_with_prefix` 为主。历史上的 `legacy_plain_with_prefix` 只适合作为局部对照，不应再被当成最终口径，否则会造成：

- train/eval prompt 形态不一致
- 文档与生产评测路径不一致
- 指标不可比较

## 5. Full-finetune 资源成本仍然很高

当前 full-finetune 在双卡下依然有很高的显存和时间成本。即使 pipeline 已经支持无 checkpoint live eval，这也只解决了磁盘压力，没有解决：

- backbone 训练成本高
- DDP 额外开销仍在
- 大规模超参搜索代价过高

因此如果后续目标是持续迭代指标，仍应认真评估更轻量的适配方式。

## 6. `find_unused_parameters=True` 仍带来额外开销

当前 DDP 仍保留了比较保守的 `find_unused_parameters=True`。这不会直接导致错误，但会带来额外 autograd 遍历和性能损耗。它不是 blocker，但属于长期应清理的训练效率问题。

## 7. Live eval 减少了磁盘压力，但提高了文档同步要求

训练后直接复用内存中的 `system` 做评测，已经是当前版本的重要事实。它的好处是：

- 不必强制保存 `final_model.pt`
- probe 运行更轻

但它也意味着：

- README 不能再把 checkpoint 当成必有产物
- 结果解释必须看 `eval_results*.json`
- 文档一旦不更新，就会很快和运行事实脱节
