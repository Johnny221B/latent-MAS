# Ours Debug And Rerun Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the `ours` inference/training observability issues, validate behavior on a 16-sample debug loop, and rerun `ours` train/eval only.

**Architecture:** First align terminal inference with a controllable prompt path, then make step-level training logging trustworthy, then run a minimal 16-sample train/eval loop to classify whether the failure was dominated by inference formatting or training quality.

**Tech Stack:** Python, PyTorch, Transformers, wandb, uv, shell wrappers, pytest.

---

### Task 1: Make `ours` Eval Streaming And Parameter-Accurate

**Files:**
- Modify: `src/cli/evaluate.py`
- Test: `tests/test_evaluate_streaming.py`
- Test: `tests/test_dag_executor.py`

**Step 1: Keep per-rank streaming outputs**

Maintain per-rank `jsonl` and `partial.json` files during eval.

**Step 2: Ensure `max_new_tokens` reaches terminal generation**

Pass the CLI value through `evaluate.py -> MultiAgentSystem -> DAGExecutor -> Agent.generate_answer(...)`.

**Step 3: Verify tests**

Run:
- `uv run --python .venv/bin/python -m pytest -q tests/test_evaluate_streaming.py`
- `uv run --python .venv/bin/python -m pytest -q tests/test_dag_executor.py -k max_new_tokens`

Expected: pass.

### Task 2: Add Terminal Inference Modes For `ours`

**Files:**
- Modify: `src/models/agent.py`
- Modify: `src/graph/dag_executor.py`
- Modify: `src/pipeline/multi_agent_system.py`
- Modify: `src/cli/evaluate.py`
- Test: `tests/test_baseline_scripts.py` or new focused test file

**Step 1: Introduce terminal inference modes**

Support at least:
- `legacy_plain_with_prefix`
- `chat_with_prefix`
- `chat_without_prefix`

Default eval mode should be `chat_with_prefix`.

**Step 2: Implement Qwen-style chat formatting**

Use chat template for the terminal question-side text input, while still allowing latent prefix injection when the mode includes prefix.

**Step 3: Expose eval control**

Add CLI args for:
- terminal inference mode
- whether terminal prefix is used

**Step 4: Verify with a minimal unit test**

Add a test that confirms terminal chat inference uses a chat template path instead of raw plain prompt assembly.

### Task 3: Fix Step-Level Wandb Logging

**Files:**
- Modify: `src/cli/multi_train.py`
- Modify: `src/cli/train.py`
- Modify: `src/utils/reporting.py` only if needed

**Step 1: Log every optimizer step**

Emit wandb payload every optimizer step, not only sparse intervals.

**Step 2: Measure gradients before zeroing**

Compute compressor/adjacency gradient norms before `optimizer.zero_grad()`.

**Step 3: Preserve console logging**

Keep readable console logs, but do not let console frequency block wandb step-level logging.

**Step 4: Verify syntax**

Run:
- `python -m py_compile src/cli/multi_train.py src/cli/train.py src/utils/reporting.py`

Expected: no output.

### Task 4: Run A 16-Sample Debug Loop For `ours`

**Files:**
- No required repo edits if earlier tasks are complete

**Step 1: Small train**

Run `ours` training with:
- 2 GPUs
- `max_samples=16`

**Step 2: Small eval**

Run `ours` evaluation with:
- 2 GPUs
- `max_samples=16`
- default terminal mode `chat_with_prefix`

**Step 3: Inspect streaming outputs**

Check:
- no obvious repeated-question degeneration
- no fixed `generated_token_count=256`
- outputs are written incrementally

**Step 4: If still degenerate, run a contrastive eval**

Repeat 16-sample eval with `chat_without_prefix` to see whether prefix injection is the dominant source of degeneration.

### Task 5: Summarize Findings

**Files:**
- Update a doc only if requested later

**Step 1: Report which path failed first**

State whether the main issue was:
- terminal inference formatting
- prefix injection
- or training quality

**Step 2: Report artifacts**

List:
- debug training output dir
- debug eval partial/final result files
- wandb run link or run id if available

---

## Execution Notes

### Completed

- `evaluate.py` now keeps per-rank streaming outputs and correctly forwards `max_new_tokens`.
- Terminal inference now supports:
  - `legacy_plain_with_prefix`
  - `chat_with_prefix`
- `evaluate.py` exposes:
  - `--inference-mode`
  - `--no-terminal-prefix`
  - `--run-baseline`
- `train.py` and `multi_train.py` now log wandb metrics every optimizer step and measure gradient norms before `zero_grad()`.
- Added a fixed small-sample config:
  - `configs/experiments/gsm8k_3agent_debug16.yaml`

### Verification Completed

- `uv run --python .venv/bin/python -m pytest -q tests/test_dag_executor.py tests/test_baseline_scripts.py tests/test_evaluate_streaming.py`
- `uv run --python .venv/bin/python -m py_compile src/models/agent.py src/graph/dag_executor.py src/pipeline/multi_agent_system.py src/cli/evaluate.py src/cli/train.py src/cli/multi_train.py src/utils/training.py`

### Debug Run Artifacts

- Debug train output:
  - `outputs/gsm8k_qwen3-8b_debug16_20260319_070143`
- Debug W&B run:
  - `1codobph`
- Observed after switching terminal inference to `chat_with_prefix`:
  - generation stopped looking like repeated question text / looping counters
  - first 16-sample eval recovered to `14/16 = 87.5%`
  - sample outputs contained normal Qwen `<think>` + answer format

### Current Interpretation

- The dominant failure mode was the terminal inference path, not a pure training collapse.
- Training observability was also misleading before this pass because gradient norms were logged after `optimizer.zero_grad()`.
- Eval now defaults to deterministic decoding (`do_sample=False`) so repeated runs on the same checkpoint are comparable.
