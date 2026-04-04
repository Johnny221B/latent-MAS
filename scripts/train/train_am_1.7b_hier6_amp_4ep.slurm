#!/bin/bash
#SBATCH --job-name=am-r1-1.7b-hier6-4ep
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=outputs/logs/am-r1-1.7b-hier6-4ep-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=24:00:00

set -euo pipefail

cd /home/chengzhi.ucsb/code/toby/latent-MAS

eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Training AM-DeepSeek-R1 Hierarchical-6agent 1.7B + AMP, 4 epochs ==="
torchrun --master_port=29702 --nproc_per_node=2 \
  src/cli/train.py \
  --config configs/experiments/hier6/am_deepseek_r1_hier6_1.7b_amp_4ep.yaml

CKPT=$(ls -td outputs/hier6/am_deepseek_r1_1.7b_amp_4ep_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval GSM8K (full test set) ==="
torchrun --master_port=29641 --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1 \
  --batch-size 4

echo "=== ALL DONE ==="
