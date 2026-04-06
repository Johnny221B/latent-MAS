#!/bin/bash
#SBATCH --job-name=eval-arch
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/eval-arch-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=18:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl


CKPT="${CKPT:-$(ls -td outputs/arch_comparison/competition_math_sequential_4agent_1ep_bf16_think_* 2>/dev/null | head -1)}"
echo "Checkpoint: ${CKPT}"

eval_dataset() {
  local name=$1
  local out_dir=$2
  echo "=== Eval ${name} ==="
  torchrun --master_port=29761 --nproc_per_node=2 \
    src/cli/evaluate.py \
    --config ${CKPT}/config.yaml \
    --checkpoint ${CKPT}/final_model.pt \
    --eval-config configs/eval/${name}.yaml \
    --max-new-tokens 8192 \
    --output-dir ${CKPT}/${out_dir} \
    --batch-size 4 \
    --worker 2 \
  && echo "=== ${name} DONE ===" \
  || echo "=== ${name} FAILED ==="
}

# eval_dataset math math
eval_dataset math500 math500
eval_dataset minerva_math minerva_math
eval_dataset aime2025 aime2025
eval_dataset amc23 amc23

echo "=== ALL DONE ==="

# arch_math_4ep/sequential_4agent_bf16_20260405_213221
# CKPT="outputs/arch_math_4ep/sequential_4agent_bf16_20260405_213221"