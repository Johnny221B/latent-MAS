#!/bin/bash
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
set -euo pipefail
cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS

CONFIG="configs/experiments/dev/arch_hierarchical_6a_4b.yaml"

# Clean outputs/dev contents before starting
rm -rf outputs/dev/*
mkdir -p outputs/dev

export MASTER_PORT=$((29500 + RANDOM % 1000))

# Step 1: Train on competition_math (all training data)
echo "========== Training on competition_math (all training data) =========="
uv run --python .venv/bin/python torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node=2 \
  src/cli/train.py \
  --config "$CONFIG"

# Find the output dir containing final_model.pt
CKPT_DIR=$(find outputs/dev -maxdepth 2 -name 'final_model.pt' -printf '%h\n' 2>/dev/null | head -1)
if [ -z "$CKPT_DIR" ]; then
  echo "ERROR: No final_model.pt found under outputs/dev/"
  ls -R outputs/dev/
  exit 1
fi
echo "Training output: $CKPT_DIR"

# Step 2: Evaluate on gsm8k (100 samples, batch_size=4, worker=2, max_new_tokens=8192)
echo "========== Evaluating on gsm8k (100 samples) =========="
export MASTER_PORT=$((29500 + RANDOM % 1000))
uv run --python .venv/bin/python torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node=2 \
  src/cli/evaluate.py \
  --config "${CKPT_DIR}/config.yaml" \
  --checkpoint "${CKPT_DIR}/final_model.pt" \
  --eval-config configs/eval/gsm8k.yaml \
  --max_samples 100 \
  --max-new-tokens 8192 \
  --output-dir "${CKPT_DIR}/gsm8k_eval" \
  --batch-size 4 \
  --worker 2

echo "========== Done =========="
