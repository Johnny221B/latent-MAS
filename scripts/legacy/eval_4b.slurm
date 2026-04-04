#!/bin/bash
#SBATCH --job-name=graph-v2-4b
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/graph-v2-4b-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=8:00:00

set -euo pipefail

cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS

eval "$(conda shell.bash hook)"
conda activate verl

echo "=== Training 4B graph_v2 (init_scale=2, adj_lr=0.05) ==="
torchrun --master_port=29688 --nproc_per_node=2 \
  src/cli/train.py --config configs/experiments/gsm8k_4agent_graph_v2.yaml

CKPT=$(ls -td outputs/graph_v2/gsm8k_qwen3-4b_* | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval 4B graph_v2 (full test set) ==="
torchrun --master_port=29633 --nproc_per_node=1 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1 \
  --batch-size 4

echo "=== Done ==="
