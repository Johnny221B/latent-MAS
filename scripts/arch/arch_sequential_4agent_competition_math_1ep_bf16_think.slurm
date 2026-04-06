#!/bin/bash
#SBATCH --job-name=comp-math-seq4-think
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/comp-math-seq4-think-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=48:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Train: Sequential 4-agent (competition_math, 1ep, Qwen3-4B, bf16, think=true) ==="
torchrun --master_port=29760 --nproc_per_node=2 \
  src/cli/train.py --config configs/experiments/arch/arch_sequential_4agent_4b_competition_math_think.yaml

CKPT=$(ls -td outputs/arch_comparison/competition_math_sequential_4agent_1ep_bf16_think_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

eval_dataset() {
  local name=$1
  local out_dir=$2
  echo "=== Eval ${name} ==="
  torchrun --master_port=29761 --nproc_per_node=2 \
    src/cli/evaluate.py \
    --config ${CKPT}/config.yaml \
    --checkpoint ${CKPT}/final_model.pt \
    --eval-config configs/eval/${name}.yaml \
    --max-new-tokens 8192 \
    --output-dir ${CKPT}/${out_dir} \
    --batch-size 32 \
    --worker 2 \
  && echo "=== ${name} DONE ===" \
  || echo "=== ${name} FAILED ==="
}

eval_dataset gsm8k gsm8k_eval
eval_dataset humaneval humaneval
eval_dataset arc_easy arc_easy
eval_dataset arc_challenge arc_challenge
eval_dataset math500 math500
eval_dataset minerva_math minerva_math

echo "=== ALL DONE ==="
