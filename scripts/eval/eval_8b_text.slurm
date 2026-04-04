#!/bin/bash
#SBATCH --job-name=eval-8b-text
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/eval-8b-text-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=8:00:00

set -euo pipefail

cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS

eval "$(conda shell.bash hook)"
conda activate verl

CKPT="outputs/graph_v2/gsm8k_qwen3-8b_20260402_031251"

echo "=== Eval 8B graph_v2 text mode (full test set) ==="
python -u src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir ${CKPT}/gsm8k_eval_text \
  --max_samples -1 \
  --batch-size 8 \
  --inference-mode chat_with_text

echo "=== Done ==="
