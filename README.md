# Latent Multi-Agent Communication Framework

基于 latent communication 的多智能体协作框架，包含：
- `ours`：可训练的 latent multi-agent system
- `single-model baseline`：通过 vLLM OpenAI API 调用单模型
- `paper LatentMAS baseline`：原论文方法的对照实现

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

默认直接使用 Hugging Face 模型名下载与缓存模型，不依赖本地 `weights/` 目录。  
如果你想显式指定缓存目录，可以在运行前设置：

```bash
export HF_HOME=.cache/huggingface
```

如果需要 wandb，上报会优先从仓库根目录 `.env` 中读取：

```bash
WANDB_API_KEY=your_key_here
```

## Training

训练入口只保留一个脚本：

```bash
bash scripts/train.sh
```

默认配置是：

```bash
configs/experiments/gsm8k_5agent.yaml
```

两卡训练示例：

```bash
NPROC_PER_NODE=2 bash scripts/train.sh
```

指定配置文件：

```bash
NPROC_PER_NODE=2 bash scripts/train.sh --config configs/experiments/gsm8k_5agent.yaml
```

训练输出会写到 `outputs/<timestamp_dir>/`，目录中通常包含：
- `config.yaml`
- `loss_log.csv`
- `eval_results*.json`（如果启用 post-train live eval）

当 `training.save_final_checkpoint=true` 时，目录中还会包含：
- `final_model.pt`

## Evaluate Ours

如果你使用独立 checkpoint eval，`ours` 的评测脚本会读取目录里的：
- `config.yaml`
- `final_model.pt`

如果你使用训练后的 live eval，则结果会直接写到 output 目录，不要求存在 `final_model.pt`。

示例：

```bash
CKPT_FOLDER=outputs/your_checkpoint_dir bash scripts/evaluate.sh
```

默认评测模式是：
- `inference-mode = chat_with_prefix`
- `max_samples = -1`

评测输出会持续刷新两个文件：
- `eval_results.json`
- `agent_logs.json`

其中 `agent_logs.json` 会记录每个样本下各 agent 的结构化日志，包括：
- system prompt
- 是否接收到 upstream latent prefix
- latent trajectory / compressed prefix 的 shape 与 norm 统计
- 终端 agent 的生成结果与 generation metadata

## Evaluate Single-Model Baseline

single-model baseline 通过 vLLM 的 OpenAI-compatible API 运行。

如果你想手动先起服务：

```bash
uv run --python .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --api-key EMPTY
```

然后在另一个终端执行评测：

```bash
uv run --python .venv/bin/python src/cli/run_baseline_single_model.py \
  --model-name Qwen/Qwen3-8B \
  --max-samples 100 \
  --max-new-tokens 2048 \
  --worker-count 8 \
  --base-url http://127.0.0.1:8000 \
  --api-key EMPTY \
  --reuse-server \
  --output-dir outputs/baselines
```

也可以直接参考：

```bash
bash scripts/eval_single.sh
```

## Evaluate Paper LatentMAS Baseline

```bash
bash scripts/eval_paper.sh
```

默认会输出到：

```bash
outputs/baselines/
```

## Project Structure

- `configs/`：角色定义、图结构先验、实验配置
- `docs/`：主文档、参考资料、records、plans
- `docs/plans/`：调试和执行计划文档
- `src/cli/`：训练、评测、baseline 的 Python 入口
- `src/models/`：基础模型封装、agent、compressor
- `src/graph/`：可学习邻接矩阵、DAG 执行器
- `src/communication/`：latent message aggregation
- `src/losses/`：任务损失与图正则
- `src/pipeline/`：`MultiAgentSystem`
- `scripts/`：仅保留 shell 启动脚本
- `.venv/`：本地虚拟环境
- `outputs/`：训练与评测结果
