#!/bin/bash
# RQ3: Performance–Cost Trade-off Experiment
# Train 3 methods on limo (119 samples), eval on GSM8K + MATH500 with varying max_new_tokens
#
# Usage:
#   bash scripts/rq3_run.sh [train|eval|all] [method]
#   method: latent | pure_prefix | single_prefix | all (default: all)
#
# Examples:
#   bash scripts/rq3_run.sh train all        # Train all 3 methods
#   bash scripts/rq3_run.sh eval latent      # Eval latent only
#   bash scripts/rq3_run.sh all              # Train + eval everything

set -e

STAGE=${1:-all}
METHOD=${2:-all}

MAX_TOKENS_LIST="8192 4096 2048 1024 512"
EVAL_TASKS="gsm8k math500"
NUM_GPUS=${NUM_GPUS:-1}

# ── Config paths ──
LATENT_CFG="configs/experiments/rq3/latent_limo_4b.yaml"
PURE_PREFIX_CFG="configs/experiments/rq3/pure_prefix_limo_4b.yaml"
SINGLE_PREFIX_CFG="configs/experiments/rq3/single_prefix_limo_4b.yaml"

# ── Output dirs (match config output.dir) ──
LATENT_OUT="outputs/rq3/latent_limo_4b"
PURE_PREFIX_OUT="outputs/rq3/pure_prefix_limo_4b"
SINGLE_PREFIX_OUT="outputs/rq3/single_prefix_limo_4b"

train_method() {
    local cfg=$1
    local name=$2
    echo "=========================================="
    echo "  Training: $name"
    echo "  Config:   $cfg"
    echo "=========================================="
    python src/cli/train.py --config "$cfg"
}

eval_method() {
    local out_dir=$1
    local name=$2
    local cfg="$out_dir/config.yaml"
    local ckpt="$out_dir/final_model.pt"

    if [ ! -f "$ckpt" ]; then
        echo "SKIP eval $name: checkpoint not found at $ckpt"
        return
    fi

    for max_tokens in $MAX_TOKENS_LIST; do
        for task in $EVAL_TASKS; do
            local eval_dir="$out_dir/eval_${task}_t${max_tokens}"
            echo "------------------------------------------"
            echo "  Eval: $name | $task | max_tokens=$max_tokens"
            echo "  Output: $eval_dir"
            echo "------------------------------------------"
            python src/cli/evaluate.py \
                --config "$cfg" \
                --checkpoint "$ckpt" \
                --eval-config "configs/eval/${task}.yaml" \
                --max-new-tokens "$max_tokens" \
                --output-dir "$eval_dir" \
                --num-gpus "$NUM_GPUS" \
                --no-thinking
        done
    done
}

# ── Train ──
if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
    if [ "$METHOD" = "latent" ] || [ "$METHOD" = "all" ]; then
        train_method "$LATENT_CFG" "Latent (ours)"
    fi
    if [ "$METHOD" = "pure_prefix" ] || [ "$METHOD" = "all" ]; then
        train_method "$PURE_PREFIX_CFG" "Pure Prefix (baseline 1)"
    fi
    if [ "$METHOD" = "single_prefix" ] || [ "$METHOD" = "all" ]; then
        train_method "$SINGLE_PREFIX_CFG" "Single Agent + Prefix (baseline 2)"
    fi
fi

# ── Eval ──
if [ "$STAGE" = "eval" ] || [ "$STAGE" = "all" ]; then
    if [ "$METHOD" = "latent" ] || [ "$METHOD" = "all" ]; then
        eval_method "$LATENT_OUT" "Latent (ours)"
    fi
    if [ "$METHOD" = "pure_prefix" ] || [ "$METHOD" = "all" ]; then
        eval_method "$PURE_PREFIX_OUT" "Pure Prefix (baseline 1)"
    fi
    if [ "$METHOD" = "single_prefix" ] || [ "$METHOD" = "all" ]; then
        eval_method "$SINGLE_PREFIX_OUT" "Single Agent + Prefix (baseline 2)"
    fi
fi

echo ""
echo "=========================================="
echo "  RQ3 experiment complete!"
echo "=========================================="
