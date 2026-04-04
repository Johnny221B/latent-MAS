#!/bin/bash
# Clean up output directories that have no trained weights (final_model.pt).
# Usage:
#   bash scripts/cleanup_outputs.sh          # dry-run (default)
#   bash scripts/cleanup_outputs.sh --delete  # actually delete

set -euo pipefail

OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
DRY_RUN=true

if [[ "${1:-}" == "--delete" ]]; then
    DRY_RUN=false
fi

if [[ ! -d "$OUTPUTS_DIR" ]]; then
    echo "Error: outputs directory '$OUTPUTS_DIR' not found."
    exit 1
fi

total_size=0
count=0

# Find all leaf experiment directories (contain config.yaml but no final_model.pt)
while IFS= read -r dir; do
    # Skip non-experiment directories (logs, baselines, etc.)
    [[ ! -f "$dir/config.yaml" && ! -f "$dir/run_provenance.json" ]] && continue

    # Skip if it has weights
    [[ -f "$dir/final_model.pt" ]] && continue

    # Skip if currently being trained (check for active slurm job writing here)
    # A running job typically has a wandb/run-* directory with recent writes
    if [[ -d "$dir/wandb" ]]; then
        latest_write=$(find "$dir/wandb" -type f -newer "$dir/config.yaml" -mmin -30 2>/dev/null | head -1)
        if [[ -n "$latest_write" ]]; then
            echo "[SKIP - active] $dir"
            continue
        fi
    fi

    dir_size=$(du -sm "$dir" 2>/dev/null | cut -f1)
    total_size=$((total_size + dir_size))
    count=$((count + 1))

    if $DRY_RUN; then
        echo "[DRY-RUN] would delete: $dir  (${dir_size}MB)"
    else
        echo "[DELETE]  $dir  (${dir_size}MB)"
        rm -rf "$dir"
    fi
done < <(find "$OUTPUTS_DIR" -type d -mindepth 1 -maxdepth 3 | sort)

echo ""
echo "Summary: $count directories without weights, total ${total_size}MB"
if $DRY_RUN; then
    echo "Run with --delete to actually remove them."
fi
