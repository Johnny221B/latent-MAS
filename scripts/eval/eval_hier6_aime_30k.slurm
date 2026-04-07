#!/bin/bash
#SBATCH --job-name=eval-hier6-aime-30k
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/eval-hier6-aime-30k-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=48:00:00

set -uo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

eval_ckpt() {
  local ckpt=$1
  local suffix=$2
  echo "=== Eval aime2025 [${suffix}] ==="
  torchrun --master_port=29770 --nproc_per_node=1 \
    src/cli/evaluate.py \
    --config ${ckpt}/config.yaml \
    --checkpoint ${ckpt}/final_model.pt \
    --eval-config configs/eval/aime2025.yaml \
    --max-new-tokens 30000 \
    --output-dir ${ckpt}/aime2025_30k \
    --batch-size 1 \
    --worker 2 \
  && echo "=== aime2025 [${suffix}] DONE ===" \
  || echo "=== aime2025 [${suffix}] FAILED ==="

  echo "=== Eval aime2026 [${suffix}] ==="
  torchrun --master_port=29770 --nproc_per_node=1 \
    src/cli/evaluate.py \
    --config ${ckpt}/config.yaml \
    --checkpoint ${ckpt}/final_model.pt \
    --eval-config configs/eval/aime2026.yaml \
    --max-new-tokens 30000 \
    --output-dir ${ckpt}/aime2026_30k \
    --batch-size 1 \
    --worker 2 \
  && echo "=== aime2026 [${suffix}] DONE ===" \
  || echo "=== aime2026 [${suffix}] FAILED ==="
}

eval_ckpt "outputs/arch_comparison/competition_math_sequential_4agent_1ep_bf16_think_20260405_134422" "seq4-1ep"

echo "=== ALL DONE ==="
