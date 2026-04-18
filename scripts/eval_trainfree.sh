#!/bin/bash
# Training-free eval: build system with random weights, directly evaluate
# Usage: bash scripts/eval_trainfree.sh
set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
source .venv/bin/activate

CONFIG=configs/experiments/rq3/latent_limo_4b_think_trainfree.yaml
OUT_DIR=outputs/rq3/latent_limo_4b_think_trainfree

# Step 1: "train" with 0 epochs to get a checkpoint with random init weights
echo "=== Building system (0 epochs) ==="
python src/cli/train.py --config $CONFIG

CKPT=$(ls -td ${OUT_DIR}_* 2>/dev/null | head -1)
echo "Checkpoint: ${CKPT}"

# Step 2: Evaluate
for max_tokens in 8192 4096 2048 1024 512; do
    for task in gsm8k math500; do
        eval_dir="${CKPT}/eval_${task}_t${max_tokens}"
        echo "=== ${task} | max_tokens=${max_tokens} ==="
        python src/cli/evaluate.py \
            --config "${CKPT}/config.yaml" \
            --checkpoint "${CKPT}/final_model.pt" \
            --eval-config "configs/eval/${task}.yaml" \
            --max-new-tokens "${max_tokens}" \
            --output-dir "${eval_dir}" \
            --num-gpus ${NUM_GPUS:-1} \
            --batch-size 4 \
            --dtype bfloat16 \
            --max_samples -1
    done
done

echo "=== TRAINING-FREE EVAL DONE ==="
