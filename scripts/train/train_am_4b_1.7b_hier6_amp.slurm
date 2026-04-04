#!/bin/bash
#SBATCH --job-name=am-4b+1.7b-hier6
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=outputs/logs/am-4b+1.7b-hier6-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=24:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Training Heterogeneous 4B+1.7B Hierarchical-6agent + AMP ==="
echo "  planner(4B) -> analyst/critic/verifier(1.7B) -> refiner(4B) -> solver(4B)"
torchrun --master_port=29701 --nproc_per_node=2 \
  src/cli/train.py \
  --config configs/experiments/hier6/am_deepseek_r1_hier6_4b_1.7b_amp.yaml

CKPT=$(ls -td outputs/hier6/am_deepseek_r1_4b_1.7b_amp_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval GSM8K ==="
torchrun --master_port=29640 --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1 --batch-size 4

echo "=== ALL DONE ==="
