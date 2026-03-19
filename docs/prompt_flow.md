# Prompt 使用位置说明

## 1. 这份文档的目的

这份文档专门回答一个工程问题：当前仓库里，`multi-agent` 系统每一步使用的 prompt 放在哪里，训练与评测各自如何使用这些 prompt，以及两个 baseline 的 prompt 又放在哪里。

本文档不讨论 latent communication 的训练原理，重点是把 prompt 的定义位置、加载链路和执行路径讲清楚，方便后续快速改 prompt 或检查实验设置。

## 2. `ours` 方法的 prompt 放在哪里

当前 `ours` 方法中，每个 agent 的角色 prompt 都定义在 `configs/roles/` 目录下，每个角色对应一个 JSON 文件，核心字段为 `system_prompt`。

当前仓库中的角色文件包括：

- [reader.json](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/roles/reader.json)
- [planner.json](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/roles/planner.json)
- [analyst.json](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/roles/analyst.json)
- [solver.json](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/roles/solver.json)
- [summarizer.json](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/roles/summarizer.json)
- [critic.json](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/roles/critic.json)

这些文件的结构基本一致，例如：

```json
{
  "role_name": "planner",
  "system_prompt": "You are a planning agent. Analyze the problem and break it down into clear, logical steps. Identify what information is needed and outline a solution strategy.",
  "reasoning_steps": 25,
  "compress_last_k": 25
}
```

其中：

- `role_name` 定义角色名称
- `system_prompt` 定义该 agent 的角色提示词
- `reasoning_steps` 定义该 agent 的 latent reasoning 步数
- `compress_last_k` 定义压缩时使用的尾部 latent states 长度

## 3. step 顺序放在哪里

prompt 内容和执行顺序是分开定义的。

角色 prompt 放在 `configs/roles/`，而多智能体的执行顺序和图结构放在 `configs/graphs/`。

当前图配置包括：

- [3agent_sequential.json](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/graphs/3agent_sequential.json)
- [5agent_fan_in.json](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/configs/graphs/5agent_fan_in.json)

例如三智能体顺序图中：

```json
{
  "agents": ["planner", "solver", "critic"],
  "adjacency_prior": [
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
  ],
  "terminal_agent_index": 2
}
```

这里定义了：

- 第 1 步是 `planner`
- 第 2 步是 `solver`
- 第 3 步是 `critic`
- 最终负责输出答案的是 `terminal_agent_index = 2`，即 `critic`

因此，一个 step 用什么 prompt，首先由图中的角色名决定，然后再去 `configs/roles/{role}.json` 中找到对应的 `system_prompt`。

## 4. `ours` 方法的 prompt 如何被加载

`ours` 的 prompt 加载入口在 [multi_agent_system.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/pipeline/multi_agent_system.py)。

整体流程如下：

1. 读取实验配置中的 graph config 路径。
2. 读取 graph config 里的 `agents` 列表。
3. 对于每个角色名，到 `roles_dir` 下读取对应 JSON。
4. 用角色配置构造 `Agent` 对象。

代码层面的关键逻辑是：

- `graph_config = json.load(...)`
- `self.agent_roles = graph_config["agents"]`
- `role_path = roles_dir / f"{role_name}.json"`
- `role_config = json.load(...)`
- `agent = Agent(..., role_config=role_config, ...)`

也就是说，`MultiAgentSystem` 本身并不硬编码 prompt 内容，它只是根据图配置去装配不同角色的 agent。

## 5. `Agent` 如何把 prompt 拼进输入

真正把 prompt 拼到模型输入中的地方在 [agent.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/models/agent.py)。

关键字段与函数如下：

- `self.system_prompt = role_config["system_prompt"]`
- `_get_role_token_ids()`
- `build_input_ids()`
- `forward_for_loss()`
- `generate_answer()`

具体过程是：

### 5.1 role prompt 的 tokenization

`_get_role_token_ids()` 会把 `system_prompt` tokenize，并缓存成 `self._role_tokens`。

也就是说，每个 agent 的角色 prompt 只需要分词一次，之后可以重复使用。

### 5.2 非终端 agent 的输入格式

对非终端 agent 而言，输入由两部分构成：

$$
\text{input}^{(i)} = [p_i ; x]
$$

其中：

- $p_i$ 表示第 $i$ 个 agent 的 role prompt token 序列
- $x$ 表示原始问题 token 序列

对应代码是 `build_input_ids()`：

$$
\texttt{input\_ids} = \texttt{concat}(\texttt{role\_ids}, \texttt{task\_token\_ids})
$$

如果该 agent 还有来自上游的 latent prefix，那么这个 prefix 不会以文本 token 的形式拼接，而是作为 embedding 前缀 `upstream_prefix` 注入模型。

因此更准确的输入可写为：

$$
\text{model input}^{(i)} = [z_i ; p_i ; x]
$$

其中 $z_i$ 是上游 agent 聚合得到的 latent prefix。

### 5.3 终端 agent 的训练输入格式

在训练阶段，终端 agent 使用 `forward_for_loss()`，输入形式为：

$$
[z_T ; p_T ; x ; y]
$$

其中：

- $T$ 表示终端 agent
- $z_T$ 是终端 agent 接收到的 latent prefix
- $p_T$ 是终端 agent 的 role prompt
- $x$ 是问题
- $y$ 是标准答案

这时模型通过 teacher forcing 预测答案部分 token，并在答案区域计算交叉熵损失。

### 5.4 终端 agent 的评测输入格式

在评测阶段，终端 agent 使用 `generate_answer()`，输入形式为：

$$
[z_T ; p_T ; x]
$$

此时不再拼接标准答案，而是让模型自回归生成输出文本。

### 5.5 `legacy_plain_with_prefix` 与 `chat_with_prefix` 的区别

当前评测脚本中，终端 agent 的生成路径支持两种主要输入模式：

- `legacy_plain_with_prefix`
- `chat_with_prefix`

它们的共同点是：

- 都可以使用上游 agent 聚合得到的 latent prefix
- 都会复用角色文件中的 `system_prompt`
- 都是在终端 agent 的 `generate_answer()` 中执行

它们的关键区别不在于“有没有 latent prefix”，而在于“问题侧文本输入是如何组织的”。

#### 5.5.1 `legacy_plain_with_prefix`

这是较早的纯 token 拼接路径。终端 agent 会先把：

- role prompt token
- question token

直接拼成：

$$
[p_T ; x]
$$

如果启用了上游 latent communication，则真正送入模型的是：

$$
[z_T ; p_T ; x]
$$

这里的含义是：

- $z_T$：上游 agent 聚合后的 latent prefix，以 embedding 前缀形式注入
- $p_T$：终端 agent 的 role prompt token
- $x$：问题 token

可以把它画成：

```text
legacy_plain_with_prefix

upstream latent prefix:   [ z_T ]
text tokens:              [ role prompt ][ question ]
model input:              [ z_T ][ role prompt ][ question ]
```

这种方式不会走 chat template，而是把 `system_prompt` 当作普通文本 token 直接拼到问题前面。

#### 5.5.2 `chat_with_prefix`

这是当前评测默认使用的路径。它会先把终端 agent 的文本输入构造成 chat prompt，而不是直接做裸 token 拼接。

也就是说，问题侧文本先被整理成近似下面这种结构：

```text
system: <system_prompt>
user:   <question>
assistant:
```

然后再通过 tokenizer 的 chat template 渲染成实际 token 序列。

如果启用了上游 latent communication，则真正送入模型的是：

$$
[z_T ; \mathrm{Chat}(p_T, x)]
$$

其中：

- $\mathrm{Chat}(p_T, x)$ 表示由 `system_prompt` 和 `question` 共同构造出的 chat-format prompt

可以把它画成：

```text
chat_with_prefix

upstream latent prefix:   [ z_T ]
chat-formatted text:      [ system: role prompt ][ user: question ][ assistant: ]
model input:              [ z_T ][ chat-formatted text ]
```

因此，`chat_with_prefix` 不是“去掉了 role prompt”，而是把 role prompt 从“普通前缀 token”变成了“chat system message 的一部分”。

#### 5.5.3 两种模式的对比图

把终端 agent 的输入并排写出来，可以更清楚地看到差异：

```text
Mode A: legacy_plain_with_prefix
[ latent prefix ][ role prompt tokens ][ question tokens ]

Mode B: chat_with_prefix
[ latent prefix ][ chat(system=role prompt, user=question, assistant prompt) ]
```

如果进一步展开成更直观的示意：

```text
legacy_plain_with_prefix
    latent part: [ z_T ]
    text part:   [ You are a solver ... ][ If Alice has 3 apples ... ]

chat_with_prefix
    latent part: [ z_T ]
    text part:   [ <system>You are a solver ...</system>
                   <user>If Alice has 3 apples ...</user>
                   <assistant> ]
```

#### 5.5.4 为什么 `chat_with_prefix` 更适合当前 Qwen 模型

当前仓库使用的是 chat-style instruction model。对这类模型来说，输入如果符合它预训练/指令微调时习惯的 chat template，通常会比“把 system prompt 当普通文本硬拼在前面”更稳定。

因此：

- `legacy_plain_with_prefix` 更接近早期实现思路
- `chat_with_prefix` 更接近当前 chat 模型的原生使用方式

这也是为什么当前评测默认改成了 `chat_with_prefix`。

#### 5.5.5 `--no-terminal-prefix` 是什么关系

还需要特别区分一个独立开关：`--no-terminal-prefix`。

这个开关控制的是：

- 终端 agent 在评测时是否使用上游 latent prefix

它与 `legacy_plain_with_prefix` / `chat_with_prefix` 不是同一个维度。

因此可以组合成：

- `legacy_plain_with_prefix` + 使用 prefix
- `chat_with_prefix` + 使用 prefix
- `chat_with_prefix` + 不使用 prefix

最后一种可以近似写成：

```text
[ chat(system=role prompt, user=question, assistant prompt) ]
```

也就是只保留 chat-format 文本输入，不再让终端 agent 接收来自上游 agent 的 latent communication。

## 6. `eval` 有没有单独的 prompt

对 `ours` 方法来说，答案是否生成于训练还是评测，并不会切换到另一套 prompt 文件。

评测入口在 [evaluate.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/evaluate.py)。这个脚本做的事情是：

1. 读取保存下来的 config。
2. 构造 `MultiAgentSystem`。
3. 加载 checkpoint。
4. 对测试集逐条调用系统推理。

因此，`eval` 复用的还是训练时那套角色 prompt，也就是 `configs/roles/*.json` 中的 `system_prompt`。区别只在于执行路径不同：

- 训练时终端 agent 调用 `forward_for_loss()`
- 评测时终端 agent 调用 `generate_answer()`

换句话说，当前 `ours` 没有专门的 “eval prompt 文件”。训练和评测共用同一套 role prompt 定义。

## 7. `single-model baseline` 的 prompt 放在哪里

单模型 baseline 入口在 [run_baseline_single_model.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/src/cli/run_baseline_single_model.py)。

这一路径没有多智能体角色 prompt，也没有单独的 prompt 模板文件。它的做法是直接把原始问题 `question` tokenize 后送进模型生成。

因此可以把它的输入近似写成：

$$
\text{input} = [x]
$$

其中 $x$ 只是原始题目文本。

这条 baseline 的特点是简单、直接，但它与 `ours` 不共享 `configs/roles/*.json` 中的角色 prompt。

## 8. 原论文 `LatentMAS baseline` 的 prompt 放在哪里

原论文 baseline 的 prompt 位于嵌套仓库 `LatentMAS/` 中，核心文件是：

- [prompts.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/LatentMAS/prompts.py)

其中主要有两组构造函数：

- `build_agent_message_sequential_latent_mas(...)`
- `build_agent_message_hierarchical_latent_mas(...)`

这两个函数会根据：

- `role`
- `question`
- `task`
- `prompt` 模式

来生成对话格式 prompt，返回值是一个 chat message 列表，例如：

```python
[
  {"role": "system", "content": system_message},
  {"role": "user", "content": user_prompt},
]
```

也就是说，原论文 baseline 用的是显式的 chat-style prompt，而不是像 `ours` 这样把 role prompt 先 tokenize 缓存，再直接拼到 token 输入前缀中。

## 9. 原论文 `LatentMAS baseline` 如何使用这些 prompt

原论文 baseline 的执行逻辑在：

- [latent_mas.py](/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/LatentMAS/methods/latent_mas.py)

这个文件里会遍历一组默认 agent，并按角色调用：

- `build_agent_message_sequential_latent_mas(...)`
- 或 `build_agent_message_hierarchical_latent_mas(...)`

然后把得到的 chat prompt 送进模型包装器，先让非终端 agent 建立 latent KV，再让 `judger` 根据前面的 latent 信息生成最终答案。

所以原论文 baseline 的 prompt 路径可以概括为：

$$
\texttt{LatentMAS/prompts.py} \rightarrow \texttt{LatentMAS/methods/latent\_mas.py}
$$

## 10. 三种方法的 prompt 对比

当前仓库中三种主要方法的 prompt 组织方式可以概括如下：

### 10.1 `ours`

- prompt 文件位置：`configs/roles/*.json`
- step 顺序位置：`configs/graphs/*.json`
- 训练与评测是否共用 prompt：是
- prompt 注入形式：`role prompt token + question token + latent prefix embedding`

### 10.2 `single-model baseline`

- prompt 文件位置：无单独模板
- 输入形式：直接使用原始问题文本
- 是否有角色分工：无

### 10.3 `paper LatentMAS baseline`

- prompt 文件位置：`LatentMAS/prompts.py`
- prompt 形式：chat-style `system + user`
- 是否区分不同 agent：是
- 是否区分不同任务：是

## 11. 工程上最值得注意的一点

当前 `ours` 虽然也是多智能体，但它和原论文 baseline 的 prompt 机制并不相同。

原论文 `LatentMAS` 的 prompt 更接近显式的文本多角色指令，只是中间通信发生在 latent KV 层；而当前 `ours` 的设计是：

- 用较短的 role prompt 作为 agent 身份定义
- 中间不显式输出文本
- 直接把上游信息以 latent prefix embedding 的形式送入下游 agent

因此，如果后续要改实验，最常见的两个入口分别是：

- 改 `ours` 的角色定义：修改 `configs/roles/*.json`
- 改 `paper baseline` 的文本指令：修改 `LatentMAS/prompts.py`

这两者不能混为一谈。
