#!/bin/bash
#SBATCH --job-name=ablation-pure-prefix-gsm8k
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/.worktrees/toby-new/outputs/logs/ablation-pure-prefix-gsm8k-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jingxwu@unc.edu
#SBATCH --time=48:00:00

set -euo pipefail
cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/.worktrees/toby-new
source /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS/.venv/bin/activate

echo "=== Train: Ablation pure_prefix (GSM8K, 2ep, Qwen3-4B) ==="
torchrun --master_port=29762 --nproc_per_node=2 \
  src/cli/train.py --config configs/experiments/ablation_pure_prefix_gsm8k_4agent.yaml

CKPT=$(ls -td outputs/ablation_pure_prefix_gsm8k_4agent_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

echo "=== Eval GSM8K ==="
torchrun --master_port=29763 --nproc_per_node=1 \
  src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1 --batch-size 4

echo "=== DONE ==="
