#!/bin/bash
#SBATCH --job-name=rq3-pp-tf
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/rq3-pp-trainfree-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
source /home/chengzhi.ucsb/code/toby/latent-MAS/.venv/bin/activate

CONFIG=configs/experiments/rq3/pure_prefix_think_trainfree.yaml
OUT_DIR=outputs/rq3/pure_prefix_think_trainfree

# Build system with 0 epochs (saves random-init checkpoint)
echo "=== Building system (0 epochs) ==="
python src/cli/train.py --config $CONFIG

CKPT=$(ls -td ${OUT_DIR}_* 2>/dev/null | head -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT/final_model.pt" ]; then
    echo "ERROR: No checkpoint found"; exit 1
fi
echo "Checkpoint: ${CKPT}"

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
            --num-gpus 1 \
            --batch-size 1 \
            --dtype bfloat16 \
            --max_samples -1
    done
done

echo "=== PURE PREFIX TRAINING-FREE EVAL DONE ==="
