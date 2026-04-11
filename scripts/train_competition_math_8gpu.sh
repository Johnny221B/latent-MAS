#!/bin/bash
set -euo pipefail

# ── paths ──
WORKTREE="/mnt/3fs/data/yfzhang/workspace/toby/latent-MAS/.claude/worktrees/prev-commit"
PYTHON="/usr/bin/python3"
CONFIG="configs/experiments/competition_math_8gpu.yaml"
EVAL_CONFIG="configs/eval/gsm8k.yaml"

# ── env ──
export HF_DATASETS_CACHE="/mnt/3fs/data/yfzhang/cache/datasets"
export HF_HOME="/mnt/3fs/data/yfzhang/cache/huggingface"
export TRANSFORMERS_CACHE="/mnt/3fs/data/yfzhang/cache/huggingface"
export WANDB_API_KEY="7d2b127cc98f6e984dd755ac646a7ee8b02160c6"
export MASTER_PORT=$((29500 + RANDOM % 1000))
export PYTHONPATH="${WORKTREE}:${PYTHONPATH:-}"

cd "${WORKTREE}"

echo "=== Starting training: competition_math 8-GPU ==="
echo "Config: ${CONFIG}"
echo "Output: outputs/competition_math_8b_8gpu"
echo ""

# ── training ──
torchrun \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  src/cli/train.py \
  --config "${CONFIG}"

echo ""
echo "=== Training done. Starting evaluation on GSM8K ==="

CKPT_FOLDER=$(ls -td "${WORKTREE}/outputs/competition_math_8b_8gpu/"*/ 2>/dev/null | head -1)
if [ -z "${CKPT_FOLDER}" ]; then
  echo "No checkpoint found, running eval with final model..."
  CKPT_ARG=""
else
  echo "Checkpoint: ${CKPT_FOLDER}"
  CKPT_ARG="--checkpoint_path ${CKPT_FOLDER}/final_model.pt"
fi

# ── evaluation on gsm8k (single GPU) ──
${PYTHON} src/cli/evaluate.py \
  --config "${CONFIG}" \
  --eval_config "${EVAL_CONFIG}" \
  ${CKPT_ARG:-} \
  --output_dir "${WORKTREE}/outputs/competition_math_8b_8gpu/eval_gsm8k"

echo "=== Done ==="
