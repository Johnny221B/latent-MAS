#!/bin/bash
#SBATCH --job-name=eval-1.7b-full
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/eval-1.7b-full-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=12:00:00

set -euo pipefail

cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS

eval "$(conda shell.bash hook)"
conda activate verl

echo ""
echo "=========================================="
echo "  Eval: 1.7b-q16-s20-cl1_20260402_053714 (full test set)"
echo "  Time: $(date)"
echo "=========================================="

python -u src/cli/evaluate.py \
  --config "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s20-cl1_20260402_053714/config.yaml" \
  --checkpoint "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s20-cl1_20260402_053714/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s20-cl1_20260402_053714/gsm8k_eval_full" \
  --max_samples -1 \
  --batch-size 8

echo ""
echo "=========================================="
echo "  Eval: 1.7b-q16-s40-cl1_20260402_055435 (full test set)"
echo "  Time: $(date)"
echo "=========================================="

python -u src/cli/evaluate.py \
  --config "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s40-cl1_20260402_055435/config.yaml" \
  --checkpoint "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s40-cl1_20260402_055435/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s40-cl1_20260402_055435/gsm8k_eval_full" \
  --max_samples -1 \
  --batch-size 8

echo ""
echo "=========================================="
echo "  Eval: 1.7b-q16-s40-cl2_20260402_061708 (full test set)"
echo "  Time: $(date)"
echo "=========================================="

python -u src/cli/evaluate.py \
  --config "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s40-cl2_20260402_061708/config.yaml" \
  --checkpoint "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s40-cl2_20260402_061708/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s40-cl2_20260402_061708/gsm8k_eval_full" \
  --max_samples -1 \
  --batch-size 8

echo ""
echo "=========================================="
echo "  Eval: 1.7b-q16-s60-cl1_20260402_063928 (full test set)"
echo "  Time: $(date)"
echo "=========================================="

python -u src/cli/evaluate.py \
  --config "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s60-cl1_20260402_063928/config.yaml" \
  --checkpoint "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s60-cl1_20260402_063928/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q16-s60-cl1_20260402_063928/gsm8k_eval_full" \
  --max_samples -1 \
  --batch-size 8

echo ""
echo "=========================================="
echo "  Eval: 1.7b-q32-s40-cl1_20260402_070713 (full test set)"
echo "  Time: $(date)"
echo "=========================================="

python -u src/cli/evaluate.py \
  --config "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q32-s40-cl1_20260402_070713/config.yaml" \
  --checkpoint "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q32-s40-cl1_20260402_070713/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q32-s40-cl1_20260402_070713/gsm8k_eval_full" \
  --max_samples -1 \
  --batch-size 8

echo ""
echo "=========================================="
echo "  Eval: 1.7b-q32-s60-cl2_20260402_072942 (full test set)"
echo "  Time: $(date)"
echo "=========================================="

python -u src/cli/evaluate.py \
  --config "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q32-s60-cl2_20260402_072942/config.yaml" \
  --checkpoint "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q32-s60-cl2_20260402_072942/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q32-s60-cl2_20260402_072942/gsm8k_eval_full" \
  --max_samples -1 \
  --batch-size 8

echo ""
echo "=========================================="
echo "  Eval: 1.7b-q8-s20-cl1_20260402_075724 (full test set)"
echo "  Time: $(date)"
echo "=========================================="

python -u src/cli/evaluate.py \
  --config "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q8-s20-cl1_20260402_075724/config.yaml" \
  --checkpoint "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q8-s20-cl1_20260402_075724/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q8-s20-cl1_20260402_075724/gsm8k_eval_full" \
  --max_samples -1 \
  --batch-size 8

echo ""
echo "=========================================="
echo "  Eval: 1.7b-q8-s40-cl1_20260402_081452 (full test set)"
echo "  Time: $(date)"
echo "=========================================="

python -u src/cli/evaluate.py \
  --config "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q8-s40-cl1_20260402_081452/config.yaml" \
  --checkpoint "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q8-s40-cl1_20260402_081452/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir "/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/outputs/graph_v2/1.7b-q8-s40-cl1_20260402_081452/gsm8k_eval_full" \
  --max_samples -1 \
  --batch-size 8

echo "=== ALL DONE ==="
