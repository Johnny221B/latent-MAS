#!/bin/bash
#SBATCH --job-name=text-mas
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2              # 申请 2 张卡
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jingxwu@unc.edu
#SBATCH --time=12:00:00

set -euo pipefail

# ===== 路径配置 =====
PROJECT_DIR="/home/chengzhi.ucsb/code/toby/latent-MAS"
ENV_ACTIVATE="${PROJECT_DIR}/.venv/bin/activate"
RUN_PY="${PROJECT_DIR}/LatentMAS/run.py"
MODEL_NAME="Qwen/Qwen3-4B"

# 确保所有必要的目录都存在
mkdir -p "${PROJECT_DIR}/outputs/logs"
LOG_DIR="${PROJECT_DIR}/logs/text_mas_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# 激活环境
source "${ENV_ACTIVATE}"

# ===== 数据集与 GPU 对应 =====
DATASETS=("arc_easy" "gsm8k")
GPUS=(0 1)

# ===== 公共参数 =====
METHOD="text_mas"
PROMPT="sequential"
MAX_NEW_TOKENS="4096"

echo "Launching ${#DATASETS[@]} text_mas jobs..."
echo "Log directory: ${LOG_DIR}"

for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  GPU="${GPUS[$i]}"
  LOG_FILE="${LOG_DIR}/${DATASET}.log"

  echo "Start task=${DATASET} on GPU=${GPU}"

  # 1. 移除 nohup（sbatch 本身就在后台）
  # 2. 移除复杂的 bash -c 嵌套（直接运行 python 即可，除非有非常复杂的逻辑）
  
  CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 \
    python "${RUN_PY}" \
      --method ${METHOD} \
      --model_name "${MODEL_NAME}" \
      --task ${DATASET} \
      --prompt ${PROMPT} \
      --max_new_tokens ${MAX_NEW_TOKENS} \
      --generate_bs 5 > "${LOG_FILE}" 2>&1 &
done

echo "All jobs launched. Waiting for background processes to finish..."

# 核心修改：等待所有后台进程结束
wait 

echo "All text_mas tasks completed at $(date)."