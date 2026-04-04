#!/bin/bash
#SBATCH --job-name=eval-sweep
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/eval-sweep-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=8:00:00

set -euo pipefail

cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS

eval "$(conda shell.bash hook)"
conda activate verl

CHECKPOINTS=(
  "outputs/gsm8k-4b-ep2-s40-q16-cl1_20260401_200203"
  "outputs/gsm8k-4b-ep5-s40-q16-cl1_20260401_205637"
  "outputs/gsm8k-4b-ep5-s40-q16-cl2_20260402_002816"
  "outputs/gsm8k-4b-ep5-s40-q32-cl1_20260402_013529"
)

for CKPT in "${CHECKPOINTS[@]}"; do
  NAME=$(basename "$CKPT")
  echo ""
  echo "=========================================="
  echo "  Eval: $NAME"
  echo "  Time: $(date)"
  echo "=========================================="

  torchrun --master_port=29633 --nproc_per_node=2 \
    src/cli/evaluate.py \
    --config "${CKPT}/config.yaml" \
    --checkpoint "${CKPT}/final_model.pt" \
    --eval-config configs/eval/gsm8k.yaml \
    --max-new-tokens 4096 \
    --output-dir "${CKPT}/gsm8k_eval_full" \
    --max_samples -1 \
    --batch-size 8
done

echo ""
echo "=========================================="
echo "  ALL EVALS DONE"
echo "=========================================="
