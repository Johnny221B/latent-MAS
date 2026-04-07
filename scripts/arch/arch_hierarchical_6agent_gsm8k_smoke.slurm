#!/bin/bash
#SBATCH --job-name=gsm8k-hier6-smoke
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=outputs/logs/gsm8k-hier6-smoke-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=4:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Train: hierarchical_6agent SMOKE (GSM8K, 6ep, 100 samples) ==="
torchrun --master_port=29728 --nproc_per_node=2 \
  src/cli/train.py \
  --config configs/experiments/arch/arch_hierarchical_6agent_4b_gsm8k_smoke.yaml \
  --max_samples 100

CKPT=$(ls -td outputs/smoke/hierarchical_6agent_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval GSM8K (100 samples) ==="
torchrun --master_port=29729 --nproc_per_node=1 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples 100 --batch-size 4

echo "=== DONE ==="

# smoke test: init_scale=0, lambda_struct=0, lambda_sparse=0.001, lambda_concentrate=0.1
