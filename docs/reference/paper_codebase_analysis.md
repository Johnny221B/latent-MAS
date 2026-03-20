# Paper 与当前 Codebase 分析笔记

## 1. 这份文档的目的

这份文档不是完整论文翻译，也不是逐行代码注释，而是把目前已经看到的内容整理成一份“论文主张 + 当前仓库实现 + 两者差异”的分析笔记，方便后续继续对照。

当前分析主要基于：

- 本地论文文件：[paper.pdf](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/docs/reference/paper.pdf)
- arXiv 摘要页面
- 当前仓库核心实现文件

需要说明的一点是：本机当前环境里没有 `pypdf`、`PyPDF2`、`fitz` 这些 PDF 文本提取库，所以这次没有直接把整篇 PDF 文本完整抽出来逐页分析。当前能稳定确认的论文信息，主要来自 arXiv 摘要页和代码实现中对 paper 的引用式注释。

## 2. 论文主线

根据 arXiv 页面 [2511.20639](https://arxiv.org/abs/2511.20639)，这篇论文标题是：

```text
Latent Collaboration in Multi-Agent Systems
```

摘要层面的核心主张可以概括成下面几条：

### 2.1 论文要解决的问题

传统的 LLM multi-agent system 大多通过文本进行中介：

- agent A 先输出一段文字
- agent B 再读取这段文字继续推理

这种方式的问题通常包括：

- token 开销大
- 信息在文本化过程中会有损失
- 推理速度慢

### 2.2 论文提出的核心思路

论文提出的是 `LatentMAS`，核心思想不是让 agent 通过文本通信，而是直接在连续隐空间中协作。

换句话说：

- agent 不再主要通过自然语言中间结果沟通
- agent 内部会生成 latent thoughts
- 这些内部表示通过 shared latent memory 传递给别的 agent

### 2.3 论文声称的优点

摘要里明确强调了三类收益：

- 更高表达能力
- 更低复杂度
- 更少 token 消耗、更快推理

同时摘要还强调，这篇 paper 的 LatentMAS 是：

```text
an end-to-end training-free framework
```

这一点非常重要，因为它和当前仓库实现有一个核心分歧，后面会专门写。

## 3. 当前仓库实现的总体逻辑

从当前代码来看，这个仓库确实是沿着 paper 的 latent collaboration 路线在做，但已经不是 paper 摘要描述里那个“纯 training-free”的版本。

当前系统的顶层入口是 [multi_agent_system.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/pipeline/multi_agent_system.py)。

它把整个系统拆成几部分：

- `BaseModelWrapper`
- `Agent`
- `LatentCompressor`
- `LearnableAdjacency`
- `DAGExecutor`
- `TaskLoss`
- `GraphLoss`

这里已经可以看出一个很重要的事实：

- 论文强调 latent collaboration
- 当前代码把 latent collaboration 做成了“带训练目标的通信系统”

## 4. 当前代码里各个模块的作用

### 4.1 BaseModelWrapper

文件：[base_model.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/base_model.py)

它负责：

- 加载 HuggingFace causal LM
- 冻结模型参数
- 支持 prefix embedding 前缀输入
- 提供 latent reasoning 所需的基础接口

这个文件里最重要的逻辑是：

- `compute_alignment_matrix()`
- `apply_alignment()`
- `latent_reasoning()`

其中 `latent_reasoning()` 的思路是：

1. 先把正常输入编码进去
2. 取最后一层 hidden state
3. 用 alignment matrix 把 hidden state 映射回输入 embedding 空间
4. 把这个映射后的向量当作“下一个 latent token embedding”
5. 利用 KV cache 继续滚动推理

因此，这里的“思考”不是文本 token 级别的，而是连续隐空间中的一步一步 rollout。

### 4.2 Agent

文件：[agent.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/agent.py)

Agent 本身不拥有独立模型权重，它只是：

- 拿到一个角色 prompt
- 共享同一个 frozen base model
- 在不同 role 下运行 latent reasoning 或 final answer generation

当前 agent 的主要角色包括：

- `reader`
- `planner`
- `analyst`
- `solver`
- `summarizer`
- `critic`

非终端 agent 的任务是：

- 读题
- 带着自己的 role prompt 做 latent reasoning
- 把最后若干步 hidden trajectory 交给 compressor

终端 agent 的任务是：

- 接收上游 latent prefix
- 结合 role prompt + question
- 训练时对标准答案做 teacher forcing
- 推理时生成最终答案文本

### 4.3 LatentCompressor

文件：[compressor.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/compressor.py)

这个模块负责把非终端 agent 产生的变长 hidden trajectory，压缩成固定长度 prefix。

结构是一个轻量 cross-attention block：

- learnable queries
- multi-head attention
- FFN
- LayerNorm

这里的核心意义是：

- paper 里说 latent representations 可以在 agent 间传递
- 当前代码把“传递什么表示”具体实现成了一个可训练的 prefix compression 模块

也就是说，当前仓库并不是直接把全部 hidden states 原样共享，而是学习一个通信压缩器。

### 4.4 LearnableAdjacency

文件：[adjacency.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/graph/adjacency.py)

这个模块把 agent 图结构做成了可学习的 soft adjacency matrix。

它的特点是：

- 只允许上三角连边，保证 DAG
- 从 prior graph 初始化
- 用 sigmoid(logits) 得到连续边权

这代表当前系统不仅在学“消息内容”，还在学“消息路由”。

### 4.5 MessageAggregator

文件：[aggregator.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/communication/aggregator.py)

这个模块做的事很直接：

- 把所有上游 prefix 按 adjacency 权重加权求和
- 再除以总权重做归一化

这一步让图结构的梯度能直接回传。

### 4.6 DAGExecutor

文件：[dag_executor.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/graph/dag_executor.py)

它是执行器，按拓扑顺序依次运行 agent：

1. 聚合上游 prefix
2. 非终端 agent 做 latent reasoning
3. 非终端输出 trajectory，经 compressor 变成 prefix
4. 终端 agent 接收 prefix 并输出 logits 或文本答案

### 4.7 Loss

当前损失函数有两块：

- `TaskLoss`
- `GraphLoss`

`TaskLoss` 只监督最终答案 token。

`GraphLoss` 则对 adjacency 做正则：

- 惩罚不在 prior 中却被打开的新边
- 惩罚 prior 中存在却被削弱的边
- 惩罚整体图过密

这说明当前仓库是在做“带图先验的可学习 latent communication”。

## 5. 当前代码的完整推理/训练链路

如果从一个问题输入开始看，逻辑大致是：

### 5.1 输入阶段

- 读取问题文本
- tokenize
- 读取实验配置中的 graph 和 roles

### 5.2 非终端 agent 阶段

每个非终端 agent 都会：

1. 构造 `[role prompt ; question]`
2. 如果有上游 prefix，则把它拼成 embedding prefix
3. 调用 `latent_reasoning()`
4. 得到长度为 `m` 的 hidden trajectory
5. 取最后 `k` 步，送进 compressor
6. 得到固定长度 latent prefix

### 5.3 消息传递阶段

每个下游节点会：

- 收到多个上游 prefix
- 按 adjacency 权重做加权和
- 得到自己的 `upstream_prefix`

### 5.4 终端 agent 阶段

终端 agent 在训练时输入的是：

```text
[upstream_prefix ; role_prompt ; question ; answer]
```

输出 logits 后，只在答案 token 上算 loss。

在推理时，终端 agent 会：

- 读取 `[upstream_prefix ; role_prompt ; question]`
- 自回归生成最终答案文本

## 6. 它和论文的关系

如果从“理念”上说，这个仓库和论文是一致的。

一致的地方主要有：

- 都强调 latent collaboration，而不是文本中介
- 都强调多 agent 之间传递的是内部表示，不是自然语言消息
- 都把 latent thoughts 当成协作的基本载体

但如果从“实现形态”上说，这个仓库已经明显偏离 paper 摘要里的原始表述。

## 7. 它和论文的主要差异

### 7.1 论文是 training-free，当前代码是 training-enabled

这是最大的差异。

从 paper 摘要看，LatentMAS 被描述为：

- end-to-end training-free

而当前代码明确有训练过程，并且只训练两块：

- `LatentCompressor`
- `LearnableAdjacency`

也就是说，当前仓库把论文的 latent collaboration 思路，发展成了一个“可训练通信层 + 冻结基础模型”的系统。

### 7.2 论文说 shared latent working memory，当前代码实现成了 prefix + graph

论文摘要中强调的是：

- shared latent working memory

当前代码里，这个概念被工程化成了两层具体机制：

- prefix communication
- graph-based routing

也就是：

- latent memory 不再只是一个抽象共享空间
- 而是被编码成固定长度 prefix
- 再沿着一张可学习 DAG 传递

### 7.3 当前代码更强调结构学习

论文摘要里更偏方法论和推理框架。

当前代码则多了一层结构学习：

- 不仅让 agent 在 latent space 里协作
- 还要学习哪些 agent 之间应该更强地通信

这让当前仓库更像一个：

```text
latent multi-agent + learnable communication topology
```

而不是纯粹的原始 LatentMAS 复刻。

## 8. 当前代码的核心思想，用最直白的话说

如果用非常口语化但准确的方式描述，这个项目就是：

“拿一个冻结的大模型，给它挂上多个 role agent，让这些 agent 不再用文字互相聊天，而是在 hidden state 空间里先各自思考，再把思考结果压成 prefix 沿着一张可学习图往后传，最后只让最后一个 agent 输出答案，并用答案 loss 来反向塑造整条通信链路。”

这句话基本就概括了当前代码的真正逻辑。

## 9. 现在阅读这个仓库时最该抓住的几个关键点

### 9.1 不要把它理解成普通 prompt-based multi-agent

这里不是：

- agent A 输出一段自然语言
- agent B 继续读这段自然语言

这里真正传递的是 latent prefix。

### 9.2 中间 agent 的“思考”默认不可解释

中间结果主要是 hidden trajectory，不是自然语言推理链。

所以这个系统的代价是：

- 通信更高效
- 但解释性更弱

### 9.3 训练不是训练 LLM，而是训练通信层

当前训练目标不是 full finetune，而是：

- 用 frozen LLM 作为 shared reasoning engine
- 只训练 latent communication 的外层结构

### 9.4 最终 supervision 只来自终端答案

没有对中间 agent 单独做监督。

这意味着整个系统是靠最终答案质量来塑造：

- prefix 内容
- graph 结构
- 信息流方向

## 10. 当前能确认的结论

基于目前已读到的内容，可以先下一个相对稳妥的结论：

这个仓库不是简单照抄论文实现，而是在论文 latent collaboration 思想的基础上，进一步加入了：

- 可训练 compressor
- 可学习 adjacency
- 图结构正则
- teacher-forced terminal supervision

所以它更像是：

```text
paper idea + trainable communication layer + graph-structured engineering extension
```

## 11. 后续建议补充的内容

后续如果要把这份分析继续做完整，建议补三类内容：

### 11.1 论文逐节精读

等环境补上 PDF 文本提取能力后，可以按章节继续补：

- 方法章节
- 理论分析章节
- 实验设置章节
- 与当前仓库实现的一一映射

### 11.2 paper 与 code 的模块映射表

建议补一张表格：

- paper 里的概念
- 代码里的模块
- 当前是否忠实实现
- 是否做了扩展

### 11.3 当前实现的潜在偏差与 bug 风险

例如后续可以进一步审查：

- `gsm8k_3agent.yaml` 实际用了 5-agent 图
- role config 覆盖逻辑是否健壮
- agent 推理步数默认值是否合理
- DDP 训练路径是否和单卡路径行为一致

这些更像“工程复现偏差分析”，值得单独成文。
