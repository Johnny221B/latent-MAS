#!/bin/bash
#SBATCH --job-name=gsm8k-h6-4b1.7b-bf16
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/gsm8k-h6-4b1.7b-bf16-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=48:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Train: Hier6 4B+1.7B (GSM8K, 4ep, bf16) ==="
torchrun --master_port=29762 --nproc_per_node=2 \
  src/cli/train.py --config configs/experiments/arch/arch_hier6_4b_1.7b_gsm8k_4ep_bf16.yaml

CKPT=$(ls -td outputs/arch_gsm8k_4ep/hier6_4b_1.7b_bf16_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval GSM8K ==="
torchrun --master_port=29763 --nproc_per_node=1 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1 --batch-size 4

echo "=== DONE ==="
