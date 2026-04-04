#!/bin/bash
#SBATCH --job-name=eval-8b
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/eval-8b-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=4:00:00

cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS

CKPT=outputs/gsm8k_qwen3-8b_20260401_054040

eval "$(conda shell.bash hook)"
conda activate verl

torchrun --master_port=29633 --nproc_per_node=1 src/cli/evaluate.py \
  --config ${CKPT}/config.yaml \
  --checkpoint ${CKPT}/final_model.pt \
  --eval-config configs/eval/gsm8k.yaml \
  --max-new-tokens 4096 \
  --output-dir ${CKPT}/gsm8k_eval \
  --max_samples -1
