# GSM8K Rerun Training And Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-run GSM8K training and all evaluations after the sequence length change, while preventing training OOM, maximizing baseline evaluation utilization, and fixing output directory behavior.

**Architecture:** Keep the training/evaluation entrypoints unchanged at the shell level, but fix the timestamped output-dir bug in the Python training CLI, clear stale processes before launch, then execute a controlled 2-GPU 1-epoch training followed by distributed evaluations for `ours`, `single-model`, and `paper LatentMAS`.

**Tech Stack:** Python, PyTorch DDP, Hugging Face Transformers, uv, bash wrappers, pytest.

---

### Task 1: Fix Timestamped Output Directory Behavior

**Files:**
- Create: `src/utils/output_paths.py`
- Modify: `src/cli/multi_train.py`
- Test: `tests/test_output_paths.py`

**Step 1: Write the failing test**

Add a test that asserts the timestamp helper returns only the timestamped run directory path for `outputs/gsm8k_qwen3-8b`.

**Step 2: Run test to verify it fails**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_output_paths.py`

Expected: fail before helper exists.

**Step 3: Write minimal implementation**

Create `build_timestamped_output_dir(...)` and use it in `src/cli/multi_train.py`.

**Step 4: Run test to verify it passes**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_output_paths.py`

Expected: `1 passed`.

**Step 5: Verify syntax**

Run: `python -m py_compile src/utils/output_paths.py src/cli/multi_train.py`

Expected: no output.

### Task 2: Clear Stale Runtime State

**Files:**
- Modify: none

**Step 1: Inspect current processes**

Run: `ps -fu "$USER" | rg 'torchrun|src/cli|run_pipeline|train.py|evaluate.py|run_baseline|LatentMAS/run.py'`

**Step 2: Kill stale training/evaluation processes**

Run: `pkill -f 'torchrun|src/cli|run_pipeline|train.py|evaluate.py|run_baseline|LatentMAS/run.py' || true`

**Step 3: Verify no matching processes remain**

Run: `ps -fu "$USER" | rg 'torchrun|src/cli|run_pipeline|train.py|evaluate.py|run_baseline|LatentMAS/run.py'`

Expected: no active training/evaluation job rows.

**Step 4: Remove the stray empty output root if it is empty**

Run: `rmdir outputs/gsm8k_qwen3-8b`

Expected: directory removed only if empty.

### Task 3: Run Safe 2-GPU Training

**Files:**
- Modify: `configs/experiments/gsm8k_3agent.yaml` only if runtime tuning is required

**Step 1: Start from the current 2048-token config**

Confirm:
- model `Qwen/Qwen3-8B`
- `epochs: 1`
- `max_seq_len: 2048`

**Step 2: Launch training with conservative memory**

Run through shell entrypoint with 2 GPUs and `HF_HOME` set.

**Step 3: Monitor memory**

Use `nvidia-smi` during startup and the first steps.

**Step 4: If OOM occurs, reduce per-GPU batch size and retry**

Adjust only `training.batch_size`, keeping all else fixed.

**Step 5: Record final output directory**

Capture the timestamped run folder containing `config.yaml`, `loss_log.csv`, and `final_model.pt`.

### Task 4: Run Distributed Evaluations

**Files:**
- Modify: runtime args only unless a bug is found

**Step 1: Run `ours` evaluation**

Use 2 GPUs through the shell wrapper.

**Step 2: Run `single-model` baseline with aggressive batch size**

Use the largest stable batch size that fits on both GPUs.

**Step 3: Run `paper LatentMAS` baseline on 2 GPUs**

Use distributed sharding and a large generation budget.

**Step 4: Save JSON outputs**

Ensure each result JSON contains parameters and generation metadata.

### Task 5: Summarize Outputs And Risks

**Files:**
- Modify: none unless user requests docs update

**Step 1: Report final training output path**

**Step 2: Report all evaluation JSON paths**

**Step 3: Report measured accuracies**

**Step 4: Note any remaining runtime or scaling limitations**
