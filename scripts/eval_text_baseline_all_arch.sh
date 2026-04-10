#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="/mnt/3fs/data/yfzhang/workspace/toby/latent-MAS/.cache/Qwen3-4B"
OUTPUT_BASE="arch_baseline_4b/new"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
DATASETS="${DATASETS:-gsm8k amc23 aime2025 arc_easy arc_challenge}"

declare -A ARCH_GRAPHS
ARCH_GRAPHS[sequential]="configs/graphs/chain_4agent.json"
ARCH_GRAPHS[dense]="configs/graphs/dense_4agent.json"
ARCH_GRAPHS[diamond]="configs/graphs/diamond_5agent.json"
ARCH_GRAPHS[hierarchical]="configs/graphs/hierarchical_4agent.json"
ARCH_GRAPHS[two_path]="configs/graphs/two_path_4agent.json"

for ARCH in sequential dense diamond hierarchical two_path; do
  GRAPH="${ARCH_GRAPHS[$ARCH]}"
  for DS in ${DATASETS}; do
    OUTDIR="${OUTPUT_BASE}/${ARCH}/${DS}"
    echo "========== ${ARCH} / ${DS} =========="

    # Create a temp config for this architecture
    TMPCONFIG=$(mktemp /tmp/text_baseline_XXXXXX.yaml)
    cat > "${TMPCONFIG}" << YAMLEOF
model:
  name: "${MODEL}"
  frozen: true
  dtype: "bfloat16"
  enable_thinking: true
graph:
  config: "${GRAPH}"
  init_scale: 0.0
  fixed_structure: true
  freeze_topology: true
  aggregation_mode: "concat"
  roles_dir: "configs/roles"
compressor:
  num_queries: 16
  num_heads: 8
  num_layers: 1
  per_agent: false
  dropout: 0.0
training:
  task: "${DS}"
  train_strategy: "communication_only"
  input_mode: "chat_with_prefix"
  e2e_gradient: false
  batch_size: 1
  lr: 0.0
  epochs: 0
  max_seq_len: 8192
reasoning:
  steps_per_agent: 2
  compress_last_k: 1
output:
  dir: "${OUTDIR}"
evaluation:
  run_after_train: false
report:
  use_wandb: false
YAMLEOF

    mkdir -p "${OUTDIR}"
    /usr/bin/python src/cli/evaluate.py \
      --config "${TMPCONFIG}" \
      --eval-config "configs/eval/${DS}.yaml" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-gpus "${NUM_GPUS}" \
      --no-agent-logs \
      --inference-mode chat_with_text \
      --output-dir "${OUTDIR}"

    rm -f "${TMPCONFIG}"
    echo "========== Done ${ARCH} / ${DS} =========="
  done
done

echo "All text baselines complete!"
