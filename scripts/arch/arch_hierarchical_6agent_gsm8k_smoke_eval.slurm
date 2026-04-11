#!/bin/bash
#SBATCH --job-name=hier6-smoke-eval
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=outputs/logs/hier6-smoke-eval-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=2:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

CKPT=outputs/smoke/hierarchical_6agent_bf16_20260407_035502

echo "=== Eval GSM8K (100 samples) ==="
torchrun --master_port=29750 --nproc_per_node=1 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples 100 --batch-size 4

echo "=== DONE ==="
