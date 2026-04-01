#!/bin/bash
#SBATCH --job-name=latent-mas
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2              # 申请 2 张 GPU
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/job-%j.out # 确保 outputs/logs 目录已存在
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jingxwu@unc.edu
#SBATCH --time=12:00:00

set -euo pipefail

# ===== 路径配置 =====
PROJECT_DIR="/home/chengzhi.ucsb/code/toby/latent-MAS"
ENV_ACTIVATE="${PROJECT_DIR}/.venv/bin/activate"
RUN_PY="${PROJECT_DIR}/LatentMAS/run.py"
MODEL_NAME="Qwen/Qwen3-4B"

# 确保 SBATCH 输出目录和任务日志目录都存在
mkdir -p "${PROJECT_DIR}/outputs/logs"
LOG_DIR="${PROJECT_DIR}/logs/latent_mas_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# 激活环境
source "${ENV_ACTIVATE}"

# ===== 数据集与 GPU 对应 =====
DATASETS=("arc_easy" "gsm8k")
GPUS=(0 1)

# ===== 公共参数 =====
METHOD="latent_mas"
PROMPT="sequential"
MAX_NEW_TOKENS="4096"
STEPS="20"

echo "Launching ${#DATASETS[@]} jobs..."

for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  GPU="${GPUS[$i]}"
  LOG_FILE="${LOG_DIR}/${DATASET}.log"

  echo "Starting ${DATASET} on GPU ${GPU}..."

  # 直接使用 & 运行后台任务，不再需要 nohup
  python "${RUN_PY}" \
    --method ${METHOD} \
    --model_name "${MODEL_NAME}" \
    --task ${DATASET} \
    --prompt ${PROMPT} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --latent_steps ${STEPS} \
    --latent_space_realign \
    --gpu_memory_utilization 0.7 > "${LOG_FILE}" 2>&1 &
done

echo "All jobs launched. Waiting for processes to complete..."
# 关键：等待所有后台 python 进程结束
wait 

echo "All tasks finished at $(date)"