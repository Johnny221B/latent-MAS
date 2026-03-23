#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=24:00:00

cd /home/chengzhi.ucsb/code/toby/latent-MAS
bash scripts/train.sh