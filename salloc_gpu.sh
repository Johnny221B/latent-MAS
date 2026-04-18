#!/usr/bin/env bash

set -euo pipefail
salloc -p hpg-b200 \
  --nodes=1 \
  --gpus=4 \
  --time=24:00:00 \
  --cpus-per-task=4 \
  --mem=256GB \
  --mail-type=ALL \
  --mail-user=jingxwu@unc.edu \
  --job-name=dev
  
srun --pty -u bash -i
