#!/bin/bash
#SBATCH --job-name=eval-seq4-bf16
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/eval-seq4-bf16-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=4:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

CKPT=outputs/arch_gsm8k_4ep/sequential_4agent_bf16_20260404_131726

echo "=== Eval GSM8K (seq4 bf16) ==="
torchrun --master_port=29765 --nproc_per_node=1 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1 --batch-size 4

echo "=== DONE ==="
