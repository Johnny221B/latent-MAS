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

echo "=== Train + Eval: hierarchical_6agent SMOKE (GSM8K, 20ep, 100 train + 100 test) ==="
torchrun --master_port=29770 --nproc_per_node=2 \
  src/cli/train.py \
  --config configs/experiments/arch/arch_hierarchical_6agent_4b_gsm8k_smoke.yaml \
  --max_samples 100 2>&1 | tee outputs/logs/smoke_interactive.log

echo "=== DONE ==="

# full run: init_scale=0, adj_lr=0.1, lambda_sparse=0, lambda_concentrate=0.01, 10 epochs
