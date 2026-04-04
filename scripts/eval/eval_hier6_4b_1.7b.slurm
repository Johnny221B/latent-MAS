#!/bin/bash
#SBATCH --job-name=eval-4b1.7b
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/eval-4b1.7b-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=4:00:00

cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS

CKPT=outputs/hier6/am_deepseek_r1_4b_1.7b_amp_20260404_001006

eval "$(conda shell.bash hook)"
conda activate verl

# Pre-download GSM8K dataset to avoid cache issues
python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main', split='test')" 2>/dev/null || true

torchrun --master_port=29634 --nproc_per_node=1 src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1
