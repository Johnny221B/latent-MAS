# Latent-MAS 训练流程说明

## 摘要

本文档面向当前仓库的实现，对整个训练流程进行统一说明。文档采用中文叙述为主、数学公式辅助的方式，既解释系统在概念上想做什么，也解释当前代码实际上是如何完成这些计算的。重点包括四个问题：

- 训练样本是什么
- 模型前向传播如何展开
- 损失函数如何定义
- 哪些参数在训练、哪些参数被冻结

从整体上看，当前仓库实现的是一个基于冻结大语言模型的潜空间多智能体系统。系统中的多个 agent 并不通过自然语言中间结果相互通信，而是通过隐藏状态轨迹生成定长 latent prefix，再沿一张可学习的有向无环图传播。最终，系统只在终端 agent 的答案位置上施加监督，并通过这一监督反向塑造通信内容与通信结构。

## 1. 问题设定

我们考虑一个监督学习问题。训练集中的每个样本都可以写成一个二元组：

$$
(x, y)
$$

其中：

- $x$ 表示问题文本
- $y$ 表示目标答案文本

在这个基础上，系统额外给定：

- 一个冻结的基础语言模型
- 一组具有不同角色提示的 agent
- 一张带先验的通信图

模型训练的目标不是让每个 agent 生成一段显式的中间推理文本，而是让这些 agent 在隐空间中协作，最终提升终端答案的预测质量。

## 2. 数据定义

数据处理逻辑定义在 [factory.py](../src/data/factory.py) 与 [base.py](../src/data/base.py)，具体数据集逻辑按任务拆分在 [gsm8k.py](../src/data/gsm8k.py)、[arc.py](../src/data/arc.py)、[competition_math.py](../src/data/competition_math.py)、[humaneval.py](../src/data/humaneval.py)、[am_deepseek_r1_distilled.py](../src/data/am_deepseek_r1_distilled.py)。

### 2.1 当前支持的数据集

当前仓库支持以下任务：

- `gsm8k`
- `arc_easy`
- `arc_challenge`
- `humaneval`
- `competition_math`
- `am_deepseek_r1_distilled`

这些任务在代码中被统一成相同的数据接口，每条样本最终都被处理为：

```text
{
  "question": ...,
  "answer": ...
}
```

其中，GSM8K 的答案字段会经过额外抽取，只保留 `####` 之后的最终数值答案。这意味着当前训练不是在拟合完整解题过程，而是在拟合最终答案字符串。

`ARC-Easy` 与 `ARC-Challenge` 虽然在统一接口里仍然表现为 `question -> answerKey`，但这里的 `question` 不是原始裸字段。dataset 层会先把原题和候选项渲染成：

```text
<question>

Choices:
A. ...
B. ...
...
```

也就是说，当前 ARC 训练和评测都明确包含选项文本；模型学习的是“带选项的题面 -> 最终字母答案”。

`HumanEval` 是当前唯一一个代码生成任务。它在 dataset 层仍被整理成统一接口：

```text
{
  "question_id": task_id,
  "question": prompt,
  "answer": canonical_solution,
  "task_id": ...,
  "prompt": ...,
  "canonical_solution": ...,
  "test": ...,
  "entry_point": ...
}
```

其中：

- 训练时使用 `prompt -> canonical_solution`，也就是代码补全监督
- 评测时不会走字符串 exact-match，而是把生成结果写成官方 `samples.jsonl` 格式，再交给 `human_eval` harness 计算 `pass@k`
- 由于官方 `HumanEval` 只有一套题，当前仓库把它显式当作本地 `60/40` debug split 使用：`train` 取前 `60%`，`test` 取后 `40%`

这意味着当前仓库里的 `humaneval` 结果主要用于调通 train/eval 链路和做本地对比，不能直接当作完整官方 HumanEval 榜单结果。

`competition_math` 是当前新增的数学题训练任务。它来自 Hugging Face 数据集 `qwedsacf/competition_math`，当前实现只读取其唯一 `train` split，并整理成：

```text
{
  "question_id": problem,
  "question": problem,
  "answer": solution,
  "level": ...,
  "type": ...
}
```

其中训练监督直接使用完整 `solution` 文本。评测与训练期 probe 则仍会从模型生成文本中抽取最终答案，抽取时优先匹配 `\boxed{...}`，其次兼容 `final answer is ...` 这类尾部答案表达。当前不会做数学表达式的符号等价判定，相关准确率仍然基于规范化后的字符串 exact-match。

`am_deepseek_r1_distilled` 是当前新增的 assistant 全输出监督任务。它来自 Hugging Face 数据集 `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`，但当前训练阶段不会直接在线读取 Hugging Face。当前实现要求先运行 `scripts/prepare_am_deepseek_r1_distilled.py`，把 `am_0.5M` 与 `am_0.9M` 两个 subset 规范化写到本地 `data/am_deepseek_r1_distilled/train.jsonl`，然后训练 dataloader 再从该本地文件直接读取。这里已经不再调用 `datasets.load_dataset("json", ...)`，而是使用仓库内置的本地 JSONL random-access 读取器，因此不应再生成 `.hf_cache`，也不应再看到 `Generating train split` 这类 Hugging Face 建表日志。每条样本会被整理成：

```text
{
  "question_id": "am-r1-<hash>",
  "question": user_content,
  "answer": assistant_content,
  "subset": ...
}
```

其中：

- `question` 取自 `messages` 中 `role == "user"` 的文本
- `answer` 取自 `messages` 中 `role == "assistant"` 的完整文本
- 训练监督会直接使用完整 assistant 输出，其中保留 `<think>...</think><answer>...</answer>` 标签
- 当前不会把监督目标裁成单独的 `answer_content`
- 当前也不提供正式 `test` split；若需要训练走势观测，应继续使用 `training_probe`
- 若本地 `train.jsonl` 文件缺失，训练会在 dataset 加载阶段直接报错，并提示先运行预处理脚本
- `--max_samples` 仍然在 dataset 层通过 `select()` 截断本地样本，但不会触发额外的 Hugging Face 缓存构建

### 2.2 训练批次的输入形式

在 dataloader 经过 `collate_fn` 后，一个 batch 的原始内容为：

- `questions: list[str]`
- `answers: list[str]`

训练 dataloader 默认会对训练集做 shuffle。当前语义是 `training.shuffle = true` 为默认值；只有在实验配置中显式设成 `false` 时，训练样本顺序才会保持固定。单卡路径和 DDP 路径都遵循这个开关。

当前训练入口还会显式应用 `training.seed`。它默认回填为 `42`，并用于：

- Python `random`
- `torch` / `torch.cuda`
- 可用时的 `numpy`
- DDP `DistributedSampler(seed=...)`
- 单卡 shuffle dataloader 的 `generator`

这意味着在代码、配置、world size 与输入数据都不变时，训练初始化与样本顺序会尽量保持可复现。

然后，这两部分文本会分别被 tokenizer 编码为：

- 问题 token：`task_token_ids`
- 问题 mask：`task_attention_mask`
- 答案 token：`answer_ids`
- 答案 mask：`answer_mask`

因此，从训练代码角度，一个 batch 的监督信号并不是“整段提示词 + 答案”的统一序列，而是问题和答案被分开编码，之后在终端 agent 中再拼接。

当前默认实验配置 [gsm8k_5agent.yaml](../configs/experiments/gsm8k_5agent.yaml) 显式设置了 `training.input_mode = chat_with_prefix`，因此终端 agent 在训练时默认会先按 chat template 组织 `system_prompt + question`，再拼接标准答案做 teacher forcing。

`competition_math` 也沿用这一路径；不同之处只在于监督答案是完整 `solution` 文本，而不是 GSM8K 的 `####` 段落或抽取后的单个最终答案字符串。

`am_deepseek_r1_distilled` 同样沿用这一路径；不同之处在于监督答案不是“最终答案字符串”，而是 assistant 的整段输出，也就是带 `<think>` 与 `<answer>` 标签的完整响应文本。

对 ARC 而言，这里的 `question` 已经是“原题 + Choices”拼接后的文本；chat template 不会再单独处理结构化 `choices` 字段。

另外，当前评测路径额外支持 `evaluation.inference_mode = chat_with_text`。这个模式只用于推理期消融，不会改变训练；它的作用是把 agent 间通信从 latent prefix 改成文本消息，以便和原始 latent communication 做对照。

当前 `evaluate.py` 还支持单题手工推理：可直接传入 `--question "..."` 和可选 `--output-dir`。这条路径不会读取评测数据集，而是构造一条临时样本，并复用与批量评测相同的 `eval_results.json`、`agent_logs.json`、`agent_log/<role>.json` 输出文件组织。当前 `eval_results.json` 只保存逐样本预测结果，不再内嵌 agent 级日志；agent 细节单独落到 `agent_logs.json` 与 `agent_log/<role>.json`。

对 `humaneval` 而言，`evaluate.py` 的主路径与上述 answer-matching 任务不同：

- 先按 `evaluation.num_samples_per_task` 对每道题生成多个 completion
- 写出 `humaneval_samples.jsonl`
- 写出 `humaneval_problems.jsonl`
- 调用官方 `human_eval.evaluation.evaluate_functional_correctness`
- 将返回的 `pass@k` 与路径信息汇总到 `eval_results.json`

若本地没有安装 `human_eval`，或者没有启用其执行 harness，当前实现会直接报错，不会退回字符串匹配。

`competition_math` 的当前正式实验配置与上述正式评测流不同：它关闭 `evaluation.run_after_train`，并将 `training_probe.samples` 设为 `0`，因此不会默认切出训练期 probe 子集。对应地：

- 整个 `train` split 都会进入训练 dataloader
- 训练期间不会写 `probe_split.json` / `probe_history.json`

若改用 debug 配置，或手动把 `training_probe.samples` 设为正数，则会启用 in-memory probe。启用后的行为是：

- 从 `train` split 中按固定随机种子留出 `training_probe.samples` 条 probe-only 样本
- 剩余样本进入训练 dataloader
- 每隔若干个 optimizer step 跑一次 probe acc
- probe 结果写到 `probe_history.json`
- 若正式 run 启用 W&B，则按同一 `global_step` 上报 `probe/accuracy` 等指标

当前 probe 指标还会额外统计：

- `max_new_tokens_count`
- `max_new_tokens_ratio`
- `degenerate`

其中 `degenerate` 的默认判定条件是：当前 probe 中至少 `50%` 的样本以 `finish_reason = "max_new_tokens"` 结束。这一信号用于尽早发现“输出反复重复并持续打满长度上限”的退化 run。

## 3. 模型组成

顶层模块定义在 [multi_agent_system.py](../src/pipeline/multi_agent_system.py)。从训练视角出发，整个系统可以拆成五个核心部分。

### 3.1 冻结的基础语言模型

基础模型定义在 [base_model.py](../src/models/base_model.py)。

它负责：

- 加载 Hugging Face 因果语言模型
- 加载 tokenizer
- 冻结所有模型参数
- 提供支持 `prefix_embeds` 的前向与 latent reasoning 接口

设基础模型的隐藏维度为 $D$，词表大小为 $V$。那么对于一段输入，基础模型本质上提供从输入 embedding 到隐藏状态、再到 logits 的映射，但当前仓库不会更新这部分参数。

### 3.2 角色化 agent

agent 逻辑定义在 [agent.py](../src/models/agent.py)。

设系统中共有 $N$ 个 agent，记第 $j$ 个 agent 为 $a_j$。每个 agent 具有：

- 一个角色名
- 一个 system prompt
- 一个 latent reasoning 步数
- 一个压缩窗口长度

在默认 5-agent 图中，agent 顺序为：

1. `reader`
2. `planner`
3. `analyst`
4. `solver`
5. `summarizer`

这些 agent 共享同一个冻结基础模型，但由于 role prompt 不同，所以它们在函数上承担不同推理角色。

### 3.3 LatentCompressor

压缩模块定义在 [compressor.py](../src/models/compressor.py)。

设第 $j$ 个非终端 agent 的 latent trajectory 为：

$$
S_j \in \mathbb{R}^{B \times T_j \times D}
$$

其中：

- $B$ 表示 batch size
- $T_j$ 表示被保留下来的隐藏状态长度
- $D$ 表示隐藏维度

压缩器的作用是把变长轨迹映射为定长前缀：

$$
P_j = \mathrm{Compressor}(S_j) \in \mathbb{R}^{B \times L_p \times D}
$$

其中 $L_p$ 是固定的 prefix 长度，也就是配置中的 `num_queries`。

这个模块是当前仓库中第一个真正参与训练的核心组件。

### 3.4 LearnableAdjacency

图结构模块定义在 [adjacency.py](../src/graph/adjacency.py)。

设图的 raw logits 为：

$$
W \in \mathbb{R}^{N \times N}
$$

soft adjacency 定义为：

$$
A = \sigma(W)
$$

其中 $\sigma(\cdot)$ 表示 sigmoid。于是：

$$
A_{ij} \in (0,1)
$$

表示从 agent $i$ 到 agent $j$ 的通信强度。

为了保证图是 DAG，当前实现只允许上三角位置存在有效边。对角线和下三角位置会被强制屏蔽。

这个模块是当前仓库中第二个参与训练的核心组件。

### 3.5 图执行器与消息聚合器

执行器定义在 [dag_executor.py](../src/graph/dag_executor.py)，消息聚合定义在 [aggregator.py](../src/communication/aggregator.py)。

对于下游节点 $j$，若它接收到多个上游 prefix，则聚合方式为：

$$
z_j = \frac{\sum_{i < j} A_{ij} P_i}{\sum_{i < j} A_{ij} + \epsilon}
$$

其中：

- $P_i$ 表示上游 agent $i$ 的 prefix
- $A_{ij}$ 表示边权
- $\epsilon$ 是很小的数，用于数值稳定

于是 $z_j$ 就是 agent $j$ 接收到的聚合 latent message。

## 4. 前向传播流程

训练时的一次前向传播可以分成六个阶段。

## 4.1 阶段一：问题编码

给定问题文本 $x$，先通过 tokenizer 得到：

$$
X = \mathrm{Tok}(x)
$$

在代码中对应为：

- `task_token_ids`
- `task_attention_mask`

同样，答案文本 $y$ 被编码为：

$$
Y = \mathrm{Tok}(y)
$$

在代码中对应为：

- `answer_ids`
- `answer_mask`

这里需要注意：问题和答案是分开编码的。答案并不会一开始就拼进所有 agent 的输入中，而只会在终端 agent 的 teacher forcing 阶段使用。

## 4.2 阶段二：非终端 agent 的 latent reasoning

对于每个非终端 agent $a_j$，它的文本输入首先构造为：

$$
I_j = [r_j ; X]
$$

其中 $r_j$ 表示第 $j$ 个 agent 的 role prompt token 序列。

如果该 agent 有上游消息，那么其真正送入基础模型的是：

$$
[\; z_j \; ; \; r_j \; ; \; X \;]
$$

其中 $z_j$ 以 embedding prefix 的形式拼接在最前面。

然后 agent 调用 `latent_reasoning()`，执行 $m$ 步潜空间推理。设第 $t$ 步末尾的隐藏状态为 $h_t$，则 latent reasoning 的核心递推思想是：

$$
e_{t+1} = \mathrm{Align}(h_t)
$$

$$
h_{t+1} = \mathrm{LLM}(e_{t+1} \mid \text{past KV cache})
$$

也就是说：

- 先取上一时刻末端隐藏状态
- 再把它映射回输入 embedding 空间
- 把这个映射结果作为下一步 latent token 的 embedding

重复该过程后，可以得到一段隐藏状态轨迹：

$$
H_j = [h_1, h_2, \dots, h_m]
$$

其中：

$$
H_j \in \mathbb{R}^{B \times m \times D}
$$

当前代码不会把整个轨迹都发给下游，而只保留最后 $k$ 个状态：

$$
S_j = H_j[:, -k:, :]
$$

## 4.3 阶段三：trajectory 压缩为 prefix

非终端 agent 的通信输出不是文本，而是压缩后的 prefix：

$$
P_j = \mathrm{Compressor}(S_j)
$$

其中：

$$
P_j \in \mathbb{R}^{B \times L_p \times D}
$$

这一步的意义在于：把可变长度的 latent reasoning 轨迹转化为固定长度的通信表示，使得图上的信息传递既可微又结构稳定。

## 4.4 阶段四：图上传播 latent message

当执行到一个下游节点 $a_j$ 时，系统会收集所有已经生成的上游 prefix，并按照邻接矩阵进行加权聚合：

$$
z_j = \frac{\sum_{i<j} A_{ij} P_i}{\sum_{i<j} A_{ij} + \epsilon}
$$

如果该节点没有上游消息，则：

$$
z_j = \varnothing
$$

在实现里，这一层非常重要，因为边权 $A_{ij}$ 会直接乘到 prefix 上，所以最终任务损失可以反向影响图结构参数。

## 4.5 阶段五：终端 agent 的 teacher forcing

终端 agent 与非终端 agent 的行为不同。训练时，它不再继续 latent reasoning，而是直接执行一次 teacher-forced forward。

它的输入可以写成：

$$
I_{\text{term}} = [\; z_{\text{term}} \; ; \; r_{\text{term}} \; ; \; X \; ; \; Y \;]
$$

基础模型输出 logits 后，代码会把 role prompt 对应的位置切掉，只保留与问题和答案对齐的部分：

$$
\mathrm{logits} \in \mathbb{R}^{B \times (|X| + |Y|) \times V}
$$

这里：

- 前 $|X|$ 个位置对应 question
- 后 $|Y|$ 个位置对应 answer

## 4.6 阶段六：监督标签构造

标签构造定义在 [base.py](../src/data/base.py) 中的 `build_labels()`。

设问题长度为 $|X|$，答案长度为 $|Y|$。则监督标签定义为：

$$
\ell = [\underbrace{-100, -100, \dots, -100}_{|X|}, y_1, y_2, \dots, y_{|Y|}]
$$

这里 `-100` 是 PyTorch 交叉熵中的忽略标记。因此：

- question 位置不参与监督
- 只有 answer 位置参与监督

这意味着当前系统的训练目标不是“重建整段输入”，而是“在给定问题和 latent communication 的条件下，预测正确答案”。

## 5. 损失函数定义

整个训练目标由两部分组成：

- 任务损失 $L_{\text{task}}$
- 图正则损失 $L_{\text{graph}}$

### 5.1 任务损失

任务损失定义在 [task_loss.py](../src/losses/task_loss.py)。

当前使用的是标准 causal language modeling 的 shift cross-entropy。设终端 logits 为：

$$
Z \in \mathbb{R}^{B \times T \times V}
$$

标签为：

$$
\ell \in \mathbb{R}^{B \times T}
$$

则实际计算使用：

$$
L_{\text{task}} = \mathrm{CE}(Z_{[:, :-1, :]}, \ell_{[:, 1:]})
$$

并且其中所有值为 `-100` 的位置会被忽略。

从语义上说，$L_{\text{task}}$ 只约束最终答案 token 的预测质量。

### 5.2 图损失

图损失定义在 [graph_loss.py](../src/losses/graph_loss.py)。

设先验图为：

$$
A^{(0)} \in \{0,1\}^{N \times N}
$$

当前可学习图为：

$$
A \in (0,1)^{N \times N}
$$

那么图损失由三项构成。

第一项是新增边惩罚：

$$
L_{\text{add}} = \sum_{(i,j): A^{(0)}_{ij}=0} A_{ij}
$$

它惩罚那些不在先验图里、但当前被打开的边。

第二项是删边惩罚：

$$
L_{\text{drop}} = \sum_{(i,j): A^{(0)}_{ij}=1} (1 - A_{ij})
$$

它惩罚那些在先验图里存在、但当前被削弱的边。

第三项是稀疏正则：

$$
L_{\text{sparse}} = \sum_{(i,j) \in \text{valid}} A_{ij}
$$

其中 `valid` 表示所有有效上三角边。

因此总图损失为：

$$
L_{\text{graph}} = \lambda_{\text{add}} L_{\text{add}} + \lambda_{\text{drop}} L_{\text{drop}} + \lambda_{\text{sparse}} L_{\text{sparse}}
$$

### 5.3 总损失

最终训练目标定义为：

$$
L = L_{\text{task}} + L_{\text{graph}}
$$

也就是说，模型既要学会预测答案，也要在图结构上保持对先验的合理偏离，而不是完全自由地长出任意连边。

## 6. 哪些参数在训练

当前系统并不是 full finetuning，而是只训练通信层。

### 6.1 冻结参数

以下参数被冻结：

- 基础语言模型全部参数

因此，LLM 本身只是一个共享推理引擎。

### 6.2 可训练参数

优化器只接收：

- `compressor.parameters()`
- `adjacency.parameters()`

这意味着训练真正更新的是：

- 如何把 latent trajectory 压缩成 prefix
- 图上每条边的通信强度

因此当前代码可以被理解为：

$$
\text{Frozen LLM} + \text{Trainable Communication Layer}
$$

## 7. 分布式训练路径

训练入口包括：

- [train.py](../scripts/train.py)
- [multi_train.py](../scripts/multi_train.py)

其中多卡训练使用 DDP。

### 7.1 DDP 逻辑

在多卡训练中：

- `compressor` 被 DDP 包装
- `adjacency.logits` 的梯度通过手动 `all_reduce` 同步
- DDP 包装统一使用 `find_unused_parameters=False`，避免在当前稳定训练路径上额外遍历 autograd 图

这说明当前分布式实现并不是把整个系统包进 DDP，而是只对真正训练的部分进行同步。

### 7.2 小样本调试时的约束

若设：

- 每卡 batch size 为 $b$
- world size 为 $n$

那么在 `drop_last=True` 时，至少需要：

$$
\text{dataset\_size} \geq b \cdot n
$$

否则不会产生任何有效 batch。

当前仓库已经加入了显式校验逻辑，避免因为 zero-batch 情况在 epoch 结束时才触发隐藏错误。

## 8. 输出与检查点

多卡训练脚本会在输出目录中保存：

- `config.yaml`
- `run_provenance.json`
- `loss_log.csv`
- `checkpoint_step*.pt`
- `final_model.pt`

若启用了 training probe，还会额外保存：

- `probe_split.json`
- `probe_history.json`

其中 `final_model.pt` 中主要包含：

- `compressor_state`
- `adjacency_state`
- `config`

不会重新保存基础模型权重，因为基础模型来自 Hugging Face 且始终冻结。

`run_provenance.json` 用于记录本次训练的最小可追溯信息，包括：

- 训练 seed
- 启动参数 `argv`
- 当前 `rank / world_size`
- `CUDA_VISIBLE_DEVICES` 等关键环境变量
- 训练时的 git commit / branch / `git status --short` / `git diff --stat HEAD`

它的目的不是替代完整实验记录系统，而是让输出目录本身能够回答“这次 run 实际是拿什么代码、什么命令、什么 seed 跑出来的”。

## 9. 与原论文的关系

从思想上看，当前仓库与原始 Latent Collaboration 论文是一致的，因为两者都强调：

- 多 agent 在 latent space 中协作
- 中间信息不是文本消息，而是内部表示
- 终端答案质量是最终目标

但从实现层面看，当前仓库已经不是最原始的 training-free 版本，而是加入了显式可训练通信层的扩展实现。具体体现在：

- 增加了可训练压缩器
- 增加了可学习邻接矩阵
- 增加了图结构正则

因此，更准确的理解方式不是“论文原样复现”，而是：

```text
基于论文思想的、带可训练通信层的 latent multi-agent 系统
```

## 10. 总结

当前 Latent-MAS 代码库的训练流程可以概括为以下过程：

给定一个问题-答案样本 $(x, y)$，系统首先让多个非终端 agent 在冻结语言模型上执行潜空间推理，得到各自的隐藏状态轨迹；然后将这些轨迹压缩为定长 prefix，并沿着一张可学习的 DAG 进行加权传播；最后由终端 agent 在 teacher forcing 条件下预测答案 token，并通过答案交叉熵与图结构正则的联合目标来更新通信层参数。

如果用一个公式总结整个训练目标，那么就是：

$$
L = \underbrace{L_{\text{task}}}_{\text{答案监督}} + \underbrace{\lambda_{\text{add}} L_{\text{add}} + \lambda_{\text{drop}} L_{\text{drop}} + \lambda_{\text{sparse}} L_{\text{sparse}}}_{\text{图结构约束}}
$$

而如果用一句话总结当前系统的本质，那么可以写成：

$$
\text{冻结大模型} + \text{角色化 latent reasoning} + \text{可训练通信压缩} + \text{可学习图结构}
$$

这就是当前仓库训练 pipeline 的核心逻辑。
