#!/bin/bash
#SBATCH --job-name=eval-seq4-think-16k
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/eval-seq4-think-16k-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=48:00:00

set -uo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

CKPT="outputs/arch_comparison/competition_math_sequential_4agent_1ep_bf16_think_20260405_134422"
echo "Checkpoint: ${CKPT}"

eval_dataset() {
  local name=$1
  local out_dir=$2
  echo "=== Eval ${name} ==="
  torchrun --master_port=29766 --nproc_per_node=2 \
    src/cli/evaluate.py \
    --config ${CKPT}/config.yaml \
    --checkpoint ${CKPT}/final_model.pt \
    --eval-config configs/eval/${name}.yaml \
    --max-new-tokens 16384 \
    --output-dir ${CKPT}/${out_dir} \
    --batch-size 2 \
    --worker 2 \
  && echo "=== ${name} DONE ===" \
  || echo "=== ${name} FAILED ==="
}

eval_dataset amc23 amc23_16k
eval_dataset aime2025 aime2025_16k
eval_dataset aime2026 aime2026_16k

echo "=== ALL DONE ==="
