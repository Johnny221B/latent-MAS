#!/bin/bash
# Monitor completed experiments and auto-update docs/experiment_log.md
# Usage: nohup bash scripts/monitor_experiments.sh &
# Or:    tmux new -d -s monitor 'bash scripts/monitor_experiments.sh'

set -uo pipefail

cd /home/chengzhi.ucsb/code/toby/latent-MAS

LOG_FILE="docs/experiment_log.md"
STATE_FILE="outputs/.monitor_seen"
POLL_INTERVAL=120  # seconds

# Create state file if not exists
touch "$STATE_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

extract_result() {
    local eval_dir="$1"
    local results_json="$eval_dir/eval_results.json"

    if [[ ! -f "$results_json" ]]; then
        return 1
    fi

    python3 -c "
import json, sys, os, re

with open('$results_json') as f:
    data = json.load(f)

m = data.get('metrics', {})
p = data.get('parameters', {})
cfg = p.get('config', {})

accuracy = m.get('accuracy', 0)
correct = m.get('correct', 0)
total = m.get('total', 0)

# Extract model info
model_cfg = cfg.get('model', {})
model_name = model_cfg.get('name', 'unknown')
agent_models = model_cfg.get('agent_models', None)

# Resolve model short name — handle local cache paths (e.g. .cache/hub/models--Qwen--Qwen3-4B/snapshots/abc123)
def resolve_model_name(name):
    if not name:
        return 'unknown'
    # Match HF cache path: models--Org--Name
    m = re.search(r'models--(.+?)--(.+?)/', name.replace(os.sep, '/'))
    if m:
        return f'{m.group(1)}/{m.group(2)}'.split('/')[-1]
    # Match Org/Name
    return name.split('/')[-1]

model_short = resolve_model_name(model_name)

if agent_models:
    models = model_cfg.get('models', {})
    if models:
        names = sorted(set(resolve_model_name(v) for v in models.values()))
        model_short = '+'.join(names)

# Extract graph info
graph_cfg = cfg.get('graph', {})
graph_config = graph_cfg.get('config', '')
graph_short = os.path.basename(graph_config).replace('.json','') if graph_config else 'unknown'

# Extract training info
train_cfg = cfg.get('training', {})
epochs = train_cfg.get('num_epochs', '?')
batch_size = train_cfg.get('batch_size', '?')
use_amp = train_cfg.get('use_amp', False)

# Extract data info
data_cfg = cfg.get('data', {})
data_task = data_cfg.get('task', 'unknown')

# Output dir (relative)
ckpt_path = p.get('config_path', '')
output_dir = cfg.get('output', {}).get('dir', ckpt_path.rsplit('/', 1)[0] if ckpt_path else 'unknown')

print(f'accuracy={accuracy:.2f}')
print(f'correct={correct}')
print(f'total={total}')
print(f'model={model_short}')
print(f'graph={graph_short}')
print(f'epochs={epochs}')
print(f'batch_size={batch_size}')
print(f'amp={use_amp}')
print(f'data={data_task}')
print(f'output_dir={output_dir}')
print(f'time={m.get(\"time_seconds\", 0):.0f}s')
"
}

append_to_log() {
    local eval_dir="$1"
    local experiment_dir="$(dirname "$eval_dir")"
    local experiment_name="$(basename "$experiment_dir")"
    local eval_name="$(basename "$eval_dir")"

    # Parse results
    local result
    result=$(extract_result "$eval_dir") || return 1

    eval "$result"

    local date_str=$(date '+%m-%d')
    local entry="| ${date_str} | ${model} | ${graph} | ${amp} | ${epochs} | ${batch_size} | **${accuracy}%** (${correct}/${total}) | \`${experiment_name}\` |"

    log "New result: ${experiment_name}/${eval_name} → ${accuracy}% (${correct}/${total})"

    # Append to auto-generated section
    if ! grep -q "## Auto-tracked Results" "$LOG_FILE" 2>/dev/null; then
        cat >> "$LOG_FILE" << 'HEADER'

---

## Auto-tracked Results

_Auto-updated by `scripts/monitor_experiments.sh`_

| Date | Model | Graph | AMP | Epochs | BS | GSM8K | Output Dir |
|------|-------|-------|-----|--------|----|-------|-----------|
HEADER
    fi

    # Check if this experiment is already logged
    if grep -q "$experiment_name" "$LOG_FILE" 2>/dev/null; then
        log "  (already in log, skipping)"
        return 0
    fi

    echo "$entry" >> "$LOG_FILE"
    log "  → appended to $LOG_FILE"
}

log "Monitor started. Polling every ${POLL_INTERVAL}s. State: $STATE_FILE"
log "Watching for new eval_results.json under outputs/"

while true; do
    # Find all eval_results.json files
    while IFS= read -r results_file; do
        eval_dir="$(dirname "$results_file")"

        # Check if already processed
        if grep -qF "$results_file" "$STATE_FILE" 2>/dev/null; then
            continue
        fi

        # Process new result
        append_to_log "$eval_dir"

        # Mark as seen
        echo "$results_file" >> "$STATE_FILE"

    done < <(find outputs/ -name "eval_results.json" -type f 2>/dev/null)

    sleep "$POLL_INTERVAL"
done
