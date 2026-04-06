#!/bin/bash
#SBATCH --job-name=compmath-hier6-lowreg
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/compmath-hier6-lowreg-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=48:00:00

set -uo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Train: Hier6 4B+1.7B (CompMath, 2ep, bf16, think, low-reg) ==="
torchrun --master_port=29764 --nproc_per_node=2 \
  src/cli/train.py --config configs/experiments/arch/arch_hier6_4b_1.7b_compmath_2ep_bf16_think_lowreg.yaml

CKPT=$(ls -td outputs/arch_comparison/competition_math_hier6_4b_1.7b_2ep_bf16_think_lowreg_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval gsm8k ==="
torchrun --master_port=29765 --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 8192 \
  --output-dir ${CKPT}/gsm8k_eval \
  --batch-size 4 \
  --worker 2 \
&& echo "=== gsm8k DONE ===" \
|| echo "=== gsm8k FAILED ==="

echo "=== ALL DONE ==="
