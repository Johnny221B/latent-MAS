#!/bin/bash
#SBATCH --job-name=e2e-grad-6a
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=outputs/logs/e2e-grad-6a-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=24:00:00

set -euo pipefail
cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Train + Eval: E2E gradient, hierarchical 6agent, GSM8K full ==="
torchrun --master_port=29750 --nproc_per_node=2 \
  src/cli/train.py \
  --config configs/experiments/dev/arch_hierarchical_6a_4b_e2e.yaml \
  2>&1 | tee outputs/logs/e2e_grad_interactive.log

echo "=== DONE ==="
