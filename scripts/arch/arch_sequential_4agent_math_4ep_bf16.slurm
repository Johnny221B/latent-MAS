#!/bin/bash
#SBATCH --job-name=math-seq4-bf16
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/math-seq4-bf16-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jingxwu@unc.edu
#SBATCH --time=48:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Train: Sequential 4-agent (math, 4ep, Qwen3-4B, model bf16) ==="
torchrun --master_port=29760 --nproc_per_node=2 \
  src/cli/train.py --config configs/experiments/arch/arch_sequential_4agent_4b_math_4ep_bf16.yaml

CKPT=$(ls -td outputs/arch_math_4ep/sequential_4agent_bf16_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval math ==="
torchrun --master_port=29761 --nproc_per_node=1 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/math.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/math_eval \
  --max_samples -1 --batch-size 4

echo "=== DONE ==="
