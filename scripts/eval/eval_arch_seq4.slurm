#!/bin/bash
#SBATCH --job-name=eval-seq4
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=outputs/logs/eval-seq4-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=4:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

CKPT=outputs/arch_comparison/sequential_4agent_20260403_235726

# Pre-download GSM8K dataset if cache is stale
python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='test')" || true

echo "=== Eval GSM8K: sequential 4-agent ==="
torchrun --master_port=29722 --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1 --batch-size 4

echo "=== DONE ==="
