#!/bin/bash
#SBATCH --job-name=eval-1.7b
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/eval-1.7b-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=4:00:00

set -euo pipefail

cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS

eval "$(conda shell.bash hook)"
conda activate verl

CHECKPOINTS=(
  "outputs/graph_v2/1.7b-q16-s20-cl1_20260402_053714"
  "outputs/graph_v2/1.7b-q16-s40-cl1_20260402_055435"
)

for CKPT in "${CHECKPOINTS[@]}"; do
  NAME=$(basename "$CKPT")
  echo ""
  echo "=========================================="
  echo "  Eval: $NAME (100 samples)"
  echo "  Time: $(date)"
  echo "=========================================="

  python -u src/cli/evaluate.py \
    --config "${CKPT}/config.yaml" \
    --checkpoint "${CKPT}/final_model.pt" \
    --eval-config configs/eval/gsm8k.yaml \
    --max-new-tokens 4096 \
    --output-dir "${CKPT}/gsm8k_eval_100" \
    --max_samples 100 \
    --batch-size 8
done

echo "=== ALL DONE ==="
