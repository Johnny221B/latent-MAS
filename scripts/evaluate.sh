#!/bin/bash
set -euo pipefail

CKPT_FOLDER="${CKPT_FOLDER:-outputs/arch_comparison/competition_math_sequential_4agent_1ep_bf16_think_20260405_134422}"
DATASET_NAMES="${DATASET_NAMES:-math500 aime2025 amc23 minerva_math}"

for DATASET_NAME in ${DATASET_NAMES}; do
  echo "========== Evaluating ${DATASET_NAME} =========="
  uv run --python .venv/bin/python torchrun \
    --master_port="${MASTER_PORT:-29655}" \
    --nproc_per_node=2 \
    src/cli/evaluate.py \
      --config "${CKPT_FOLDER}/config.yaml" \
      --checkpoint "${CKPT_FOLDER}/final_model.pt" \
      --eval-config configs/eval/${DATASET_NAME}.yaml \
      --max-new-tokens 8192 \
      --output-dir "${CKPT_FOLDER}/${DATASET_NAME}" \
      "$@"
done
