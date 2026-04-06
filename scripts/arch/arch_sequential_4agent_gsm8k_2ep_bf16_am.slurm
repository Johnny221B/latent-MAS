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

set -uo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS

CKPT=$(ls -td outputs/arch_comparison/am_sequential_4agent_2ep_bf16_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

eval_dataset() {
  local name=$1
  local out_dir=$2
  echo "=== Eval ${name} ==="
  uv run --python .venv/bin/python torchrun --master_port=29761 --nproc_per_node=2 \
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

eval_dataset humaneval humaneval
eval_dataset arc_easy arc_easy
eval_dataset arc_challenge arc_challenge

echo "=== ALL DONE ==="
