#!/bin/bash
#SBATCH --job-name=rq3-eval-pureprefix
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/rq3-eval-pureprefix-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
source /home/chengzhi.ucsb/code/toby/latent-MAS/.venv/bin/activate

# Find latest checkpoint
CKPT=$(ls -td outputs/rq3/pure_prefix_limo_4b_* 2>/dev/null | head -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT/final_model.pt" ]; then
    echo "ERROR: No pure_prefix checkpoint found"; exit 1
fi
echo "Checkpoint: ${CKPT}"

MAX_TOKENS_LIST="8192 4096 2048 1024 512"
EVAL_TASKS="gsm8k math500"

for max_tokens in $MAX_TOKENS_LIST; do
    for task in $EVAL_TASKS; do
        eval_dir="${CKPT}/eval_${task}_t${max_tokens}"
        echo "=== Eval: pure_prefix | ${task} | max_tokens=${max_tokens} ==="
        python src/cli/evaluate.py \
            --config "${CKPT}/config.yaml" \
            --checkpoint "${CKPT}/final_model.pt" \
            --eval-config "configs/eval/${task}.yaml" \
            --max-new-tokens "${max_tokens}" \
            --output-dir "${eval_dir}" \
            --num-gpus 1 \
            --batch-size 4 \
            --dtype bfloat16 \
            --no-thinking \
            --max_samples -1
    done
done

echo "=== RQ3 Pure Prefix eval DONE ==="
