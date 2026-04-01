#!/usr/bin/env bash

set -euo pipefail
salloc -p hpg-b200 \
  --nodes=1 \
  --gpus=2 \
  --time=24:00:00 \
  --cpus-per-task=4 \
  --mem=256G \
  --mail-type=ALL \
  --mail-user=jingxwu@unc.edu \
  --job-name=toby-test
  
srun --pty -u bash -i
