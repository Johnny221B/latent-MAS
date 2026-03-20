# Method

## 问题设定

当前仓库实现的是一个 role-aware latent multi-agent system。系统包含一组按 DAG 顺序执行的 agent，每个 agent 共享同一个大语言模型 backbone，但携带不同的角色 prompt，并通过 latent prefix 而不是显式自然语言中间结果进行通信。

给定输入问题 $x$，系统在图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 上执行多步协同推理。每个节点 $i$ 对应一个 agent，每条边 $i \to j$ 表示从上游 agent $i$ 到下游 agent $j$ 的潜空间通信通道。

## 图结构与角色先验

通信强度由可学习邻接矩阵 $A \in \mathbb{R}^{n \times n}$ 表示，其中：

$$
A_{ij} \in (0,1)
$$

表示边 $i \to j$ 的软门控强度。

训练并不是从零学习一张任意图，而是从带角色先验的 `adjacency_prior` 出发。当前实现会结合：

- 图配置里的 `agents`
- `execution_order`
- `terminal_agent_index`
- `adjacency_prior`

共同确定允许的边和执行顺序。

## Agent 计算

对任意 agent $i$，输入由三部分构成：

1. 任务问题 $x$
2. 角色 prompt $p_i$
3. 来自上游节点的聚合 latent prefix $z_i$

agent 在共享 backbone 上执行 latent reasoning，得到 hidden-state trajectory：

$$
S_i = f_i(x, z_i, p_i)
$$

当前实现只保留 trajectory 的最后 `compress_last_k` 个 hidden states，再送入共享 compressor：

$$
P_i = C(S_i)
$$

其中 $P_i$ 是固定长度的 latent prefix，供所有下游节点读取。

## Latent Message Passing

对于下游节点 $j$，当前实现采用加权求和聚合：

$$
z_j = \sum_{i \in \mathcal{N}(j)} A_{ij} P_i
$$

这里的 `A_{ij}` 是 learned adjacency，`P_i` 是上游消息。聚合后的 $z_j$ 不是文本 token，而是直接作为 prefix embeddings 注入到下游 agent 的模型输入前部。

## 终端监督

非终端 agent 负责产生可传递的 latent message。真正接收任务监督的是终端 agent。

训练时，终端 agent 在 teacher forcing 下读取：

- 聚合后的 terminal prefix
- 自己的角色 prompt
- 问题文本
- 标准答案 token

并在答案区域上计算交叉熵损失。

评测时，终端 agent 只读取 prefix + prompt + question，然后自回归生成答案文本。

## 当前版本的训练策略

当前仓库不是只有一种训练口径，而是通过 config 切换两种策略：

### 1. `communication_only`

只训练：

- adjacency
- compressor

backbone 保持冻结。这个模式最接近“只学习 communication layer”的原始设定。

### 2. `full_finetune`

在训练 adjacency 与 compressor 的同时，也训练共享 backbone。

因此当前版本的 method 应该描述为：

- 方法结构始终是 latent multi-agent communication
- 具体哪些参数参与优化，取决于 experiment config

不能把所有运行都写成“当前版本只能训练 communication layer”。

## 目标函数

总损失写为：

$$
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_{\text{add}} \mathcal{L}_{\text{add}}
+
\lambda_{\text{drop}} \mathcal{L}_{\text{drop}}
+
\lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}}
$$

其中：

- $\mathcal{L}_{task}$：终端答案监督
- $\mathcal{L}_{add}$：惩罚偏离先验而新开的边
- $\mathcal{L}_{drop}$：惩罚先验边被压弱或删除
- $\mathcal{L}_{sparse}$：鼓励整体稀疏

这个目标意味着系统既在追求任务性能，也在保留图结构上的可解释偏置。

## 当前实现边界

当前版本仍有几个重要边界：

- 非终端 agent 的 latent reasoning 本身没有独立中间监督
- 所有 agent 共享同一个 backbone，而不是异构 agent
- compressor 当前是共享模块，而不是 role-pair-specific compressor
- eval 默认强调 `chat_with_prefix` 路径，而不是旧的 plain prompt 诊断路径

这些边界定义了“当前 method 的已实现版本”，不是所有可能扩展都已经落地。
