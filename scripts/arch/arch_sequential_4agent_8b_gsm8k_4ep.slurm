#!/bin/bash
#SBATCH --job-name=gsm8k-seq4-8b
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=outputs/logs/gsm8k-seq4-8b-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=48:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Train: sequential_4agent (GSM8K, 4ep, Qwen3-8B) ==="
torchrun --master_port=29740 --nproc_per_node=2 \
  src/cli/train.py --config configs/experiments/arch/arch_sequential_4agent_8b_gsm8k_4ep.yaml

CKPT=$(ls -td outputs/arch_gsm8k_4ep_8b/sequential_4agent_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval GSM8K ==="
torchrun --master_port=29741 --nproc_per_node=1 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1 --batch-size 4

echo "=== DONE ==="
