#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CKPT_FOLDER="/mnt/3fs/data/yfzhang/workspace/toby/latent-MAS/.claude/worktrees/prev-commit/outputs/dev/e2e_4b_per_agent_compressor_20260409_225327"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"

/usr/bin/python src/cli/evaluate.py \
  --config "${CKPT_FOLDER}/config.yaml" \
  --checkpoint "${CKPT_FOLDER}/final_model.pt" \
  --eval-config "configs/eval/${DATASET_NAME}.yaml" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "${CKPT_FOLDER}/${DATASET_NAME}" \
  "$@"
