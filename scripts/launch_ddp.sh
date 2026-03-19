#!/bin/bash
# Usage: bash scripts/launch_ddp.sh configs/experiments/gsm8k_3agent.yaml [max_samples]

CONFIG=$1
MAX_SAMPLES=${2:-""}
NUM_GPUS=8
MASTER_PORT=29500

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

echo "Launching $NUM_GPUS processes..."

for RANK in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$RANK \
    RANK=$RANK \
    LOCAL_RANK=0 \
    WORLD_SIZE=$NUM_GPUS \
    MASTER_ADDR=localhost \
    MASTER_PORT=$MASTER_PORT \
    python scripts/multi_train.py --config $CONFIG $MAX_SAMPLES_ARG &
    echo "  Started rank $RANK on GPU $RANK"
done

echo "All processes launched. Waiting..."
wait
echo "Done."