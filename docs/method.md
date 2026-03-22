# Method

## 问题设定

### 角色感知的多智能体潜空间通信网络

我们考虑一个有向多智能体通信网络

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E}),
$$

其中每个节点 $i \in \mathcal{V}$ 对应一个基于大语言模型的 agent，每条有向边 $i \to j$ 表示 agent $i$ 可以向 agent $j$ 发送一条 latent message。记 agent 总数为

$$
n = |\mathcal{V}|.
$$

我们使用一个可学习的邻接矩阵来表示通信结构：

$$
A \in \mathbb{R}^{n \times n},
$$

其中 $A_{ij}$ 表示从 agent $i$ 到 agent $j$ 的连接强度，或者说通信门控强度。每个 agent 都被赋予一个预定义的角色，例如 `planner`、`solver` 或 `critic`。每个角色都可以配有一个结构化的角色说明文件，例如 JSON，其中包含该角色的特定信息。

我们并不是从零开始学习整张图结构，而是先构造一个带有角色先验的邻接矩阵

$$
A^{(0)} \in \{0,1\}^{n \times n},
$$

它来源于经典多智能体系统中的角色交互模式。具体来说，

$$
A^{(0)}_{ij} =
\begin{cases}
1, & \text{如果边 } i \to j \text{ 属于某种经典角色交互模式}, \\
0, & \text{否则。}
\end{cases}
$$

这个先验图起到归纳偏置的作用：我们希望最终学到的通信图能够尽量贴近经典角色结构，但同时又允许它根据数据进行自适应调整。

### 任务执行过程

给定一个输入任务 $Task_t$，系统会在图 $\mathcal{G}$ 上执行一轮或多轮协同推理。

对于任意一个 agent $i$，它的输入由两部分组成：

1. 任务描述 $Task_t$
2. 来自上游 agent 的 latent message 聚合结果 $z_i$

在这些输入以及自身角色设定的条件下，agent $i$ 会进行 $m$ 步 latent reasoning，并产生一个内部 latent state $S_i$。

这里的 $S_i$ 表示 agent $i$ 在完成 $m$ 步推理之后的内部推理状态。一个自然的设计是，不把它定义成完整的 KV cache，而是定义成一个有限长度的 latent trajectory，或者一段 hidden-state sequence。这样做的好处是，$S_i$ 更容易被压缩，也更适合在 agent 之间传递。

形式化地，agent $i$ 的计算过程可以抽象写成：

$$
S_i = f_i(Task_t, z_i, \mathrm{role}_i),
$$

其中，$z_i$ 表示从上游节点接收到的 latent 信息聚合结果，$f_i$ 表示 agent $i$ 的 latent reasoning 过程。

### Latent Message Passing

当 agent $i$ 完成 $m$ 步推理并得到 $S_i$ 之后，它会使用一个 compressor，把内部 latent state 转换成一个固定长度的 latent prefix，再发送给下游 agent：

$$
P_i = C(S_i),
$$

其中当前实现使用的是一个共享 compressor $C$。更一般的 $C_{i \to j}$ 或 role-pair-specific compressor 仍然可以作为后续扩展方向，但不是当前版本默认能力。

对于下游 agent $j$，它会把所有来自上游节点的消息聚合成一个统一的 latent prefix：

$$
z_j = \sum_{i \in \mathcal{N}(j)} A_{ij} P_i,
$$

其中 $\mathcal{N}(j)$ 表示节点 $j$ 的所有上游邻居集合。

随后，这个聚合后的消息 $z_j$ 会作为 latent prefix 注入到 agent $j$ 的输入前部，并与任务输入和角色相关提示一起送入模型。agent $j$ 再基于这个增强后的输入继续执行自己的 latent reasoning。

这个形式化描述表达的核心思想是：agent 之间并不是通过自然语言显式交流，而是通过对内部推理过程进行压缩之后得到的连续潜空间表示来通信。

### 学习目标

当前实现中，系统真正可训练的部分是：

- 通信图 $A$
- compressor $C$

基础 backbone agent 保持冻结；非终端 agent 的 latent reasoning 负责生成可被压缩的 latent trajectory，但其参数本身不在当前版本中更新。因此，当前训练目标应理解为“学习 communication layer”，而不是“联合更新所有 agent 的内部推理参数”。

训练目标既要优化最终任务性能，也要鼓励学到的图结构保持稀疏、可解释，并尽量贴近经典角色先验图 $A^{(0)}$。

我们把总损失写成：

$$
\mathcal{L}
=
\mathcal{L}_{\text{task}}
+
\lambda_{\text{add}} \mathcal{L}_{\text{add}}
+
\lambda_{\text{drop}} \mathcal{L}_{\text{drop}},
$$

其中：

- $\mathcal{L}_{\text{task}}$ 是最终任务损失
- $\mathcal{L}_{\text{add}}$ 用来惩罚那些在先验图中不存在、但被模型新打开的边
- $\mathcal{L}_{\text{drop}}$ 用来惩罚那些在先验图中本来存在、但被模型抑制或删除的边

这个目标允许系统在有利于任务性能时偏离经典角色模式，但同时会对这种偏离施加结构性代价。

### 任务监督与评测类型

当前仓库已经同时覆盖两类任务：

- 文本答案任务，例如 `gsm8k`、`arc_*`
- 数学题文本答案任务，例如 `competition_math`
- 代码生成任务，例如 `humaneval`

二者在训练阶段都可以写成监督学习的 `(x, y)` 形式，但 `y` 的语义不同：

- 对 `gsm8k` 一类任务，`y` 是最终答案字符串
- 对 `competition_math`，`y` 是从完整题解 `solution` 中抽取出的最终答案字符串
- 对 `humaneval`，`y` 是代码补全 `canonical_solution`

二者在评测阶段则显式分叉：

- 文本答案任务使用答案抽取加 exact-match
- `competition_math` 仍属于答案抽取加 exact-match，但答案抽取优先读取 `\boxed{...}`
- `humaneval` 使用代码执行型 functional correctness，并汇总成 `pass@k`

因此，当前方法的 communication layer 是共享的，但最终任务指标不再假设所有任务都可被约化为单一字符串准确率。

对于 `competition_math`，当前默认反馈信号不是训练结束后的整套 eval，而是训练过程中的固定 `100` 条 probe 子集准确率。这个 probe 子集不参与梯度更新，作用是观察 communication layer 学到的表示是否随着 `global_step` 推进而提升答案正确率。

### 关于信息传递机制的开放设计问题

当前 formulation 中一个核心但尚未完全解决的问题是：上游 agent 和下游 agent 之间的信息究竟应该如何传递。

虽然整体框架假设 agent $i$ 会经过 $m$ 步 latent reasoning 得到一个内部状态 $S_i$，但以下几个问题仍然是开放的设计选择：

- $S_i$ 应该被具体定义成什么
- 从 $S_i$ 中应该提取什么作为通信载体
- 下游 agent $j$ 应该如何读取和利用这些信息

#### 方案一：压缩 hidden-state trajectory

一种自然的选择是，把 $S_i$ 定义为 agent $i$ 的 latent reasoning trajectory，也就是它在任务 $x$ 上进行 $m$ 步推理过程中产生的一系列内部状态。具体地，如果 agent $i$ 进行了 $m$ 步 latent reasoning，那么它的上游状态可以写成：

$$
S_i = \bigl(h_i^{(1)}, h_i^{(2)}, \dots, h_i^{(m)}\bigr).
$$

这里的每个 $h_i^{(t)}$ 不一定非要是单个向量。一个更有表达能力的设计是，让每个 step 对应一小段 latent token，因为如果每一步只用一个向量表示，容量可能会过于受限。

然后，上游 agent 使用共享 compressor，把这一段 latent trajectory 压缩成一个固定长度的 latent prefix：

$$
P_i = C(S_i) \in \mathbb{R}^{L_p \times d},
$$

其中，$C$ 是一个可学习的共享 compressor，$P_i$ 表示由上游 agent $i$ 产生、供其所有下游节点使用的 latent message。

当下游 agent $j$ 收到来自多个上游节点的 prefix 信息后，它先把这些 prefix 聚合成一个接收端 prefix，记为 $P_j^{\mathrm{recv}}$，然后把这个 prefix 拼接到自己的输入前面：

$$
\bigl[\, P_j^{\mathrm{recv}} ; X_j \,\bigr],
$$

其中，$X_j$ 表示任务输入和 agent $j$ 的角色 prompt 对应的 token 表示。然后，agent $j$ 会在这个增强输入上继续进行自己的 latent reasoning。

这种设计与 prefix tuning 的设置有较强的相似性。不同之处在于，prefix tuning 学的是一个静态 prefix 参数，而这里的 prefix 不是静态参数，而是由上游 agent 的 latent reasoning trajectory 动态生成的。因此，训练时不仅要确保下游 agent 能够学会读取并利用这些 prefix token，还必须保证在存在多个上游 agent 时，不同发送方传来的信息仍然是可区分的。

不过，这里仍然存在一个核心疑问：仅仅压缩一段 hidden-state sequence，是否真的足够支持有效的跨 agent 通信。更具体地说，当原始推理过程较为复杂时，一个较短的压缩 latent representation 是否能够稳定保留那些真正对下游 agent 有帮助的信息，这一点目前仍然并不明确。
