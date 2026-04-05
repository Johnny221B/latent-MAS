#!/bin/bash
#SBATCH --job-name=gsm8k-seq4-bf16-am
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/gsm8k-seq4-bf16-am-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=48:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Train: Sequential 4-agent (GSM8K, 2ep, Qwen3-4B, model bf16) ==="
torchrun --master_port=29760 --nproc_per_node=2 \
  src/cli/train.py --config configs/experiments/arch/arch_sequential_4agent_4b_am.yaml

CKPT=$(ls -td outputs/arch_comparison/am_sequential_4agent_2ep_bf16_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval GSM8K ==="
torchrun --master_port=29761 --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \

torchrun --master_port=29761 --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/humaneval.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/humaneval \

torchrun --master_port=29761 --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/arc_easy.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/arc_easy \

torchrun --master_port=29761 --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/arc_challenge.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/arc_challenge \

echo "=== DONE ==="
