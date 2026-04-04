#!/bin/bash
#SBATCH --job-name=evaluate-revised-prompt
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/job-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jingxwu@unc.edu
#SBATCH --time=4:00:00

cd /home/chengzhi.ucsb/code/toby/latent-MAS
bash scripts/evaluate.sh