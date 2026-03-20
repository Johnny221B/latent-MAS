# Probe64 Train Eval Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair the `probe64` training and inference path, run a `64`-sample train/eval loop with `batch_size=32`, and document each version of fixes and results without storing checkpoint files.

**Architecture:** First lock down regressions with failing tests for the two known root causes (`bf16`/compressor dtype mismatch and `full_finetune` DDP helper access), then implement the smallest fixes needed to make both training strategies runnable. After that, execute a staged experiment loop: communication-only first to measure the communication layer ceiling, then full-finetune only if the accuracy target is not met. Training must flow directly into evaluation using the live in-memory model state, while docs and experiment logs are updated after each run and only lightweight CSV/JSON artifacts are retained.

**Tech Stack:** Python, PyTorch, Transformers, torchrun/DDP, pytest, YAML configs, markdown docs.

---

### Task 1: Capture The Known Probe64 Failure Modes In Tests

**Files:**
- Modify: `tests/test_compressor.py`
- Modify: `tests/test_base_model.py`
- Modify: `tests/test_config.py`

**Step 1: Write the failing dtype-alignment test**

Add a focused test that feeds `bfloat16` latent states into the compressor and asserts the module handles mixed incoming dtype without raising a matmul mismatch.

**Step 2: Run the focused test to verify it fails for the right reason**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_compressor.py -k dtype`

Expected: FAIL on the current dtype mismatch path or missing compatibility logic.

**Step 3: Write the failing DDP-helper access test**

Add a focused test for `BaseModelWrapper` helper access that simulates a DDP-wrapped base model and verifies `get_input_embeddings()` still works.

**Step 4: Run the focused test to verify it fails**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_base_model.py -k ddp`

Expected: FAIL because helper access still assumes a raw HF model.

**Step 5: Add a config expectation test**

Add a config test that asserts the active small-probe configs are normalized around `64` samples and `batch_size=32`.

### Task 2: Fix The Root Causes With Minimal Production Changes

**Files:**
- Modify: `src/models/compressor.py`
- Modify: `src/models/base_model.py`
- Modify: `src/cli/train.py`
- Modify: `src/cli/evaluate.py`
- Modify: `src/utils/config.py` only if needed

**Step 1: Fix compressor dtype alignment**

Cast compressor inputs and masks to the module parameter dtype/device at the module boundary so a bf16 backbone can feed a float32 compressor safely.

**Step 2: Fix DDP-transparent helper access in the base wrapper**

Unwrap the underlying model consistently for helper reads such as embeddings and other direct helper calls used during latent reasoning.

**Step 3: Add a no-checkpoint train-to-eval path**

Expose a path that lets training evaluate the live `MultiAgentSystem` instance directly after training, without requiring `torch.save(...)` / `torch.load(...)` of model states.

**Step 4: Re-run the targeted tests**

Run:
- `uv run --python .venv/bin/python -m pytest -q tests/test_compressor.py -k dtype`
- `uv run --python .venv/bin/python -m pytest -q tests/test_base_model.py -k ddp`
- `uv run --python .venv/bin/python -m pytest -q tests/test_config.py`

Expected: pass.

**Step 5: Run syntax verification**

Run: `python -m py_compile src/models/compressor.py src/models/base_model.py src/cli/train.py src/cli/evaluate.py src/utils/config.py`

Expected: no output.

### Task 3: Keep Probe64 Configs And Docs Aligned

**Files:**
- Modify: `configs/experiments/gsm8k_5agent_probe64_comm_only.yaml`
- Modify: `configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml`
- Modify: `docs/current_version_issues_2026-03-19.md`

**Step 1: Confirm the probe configs match the active contract**

Keep:
- `batch_size: 32`
- `drop_last: false`
- `train_probe_samples: 64`
- `test_probe_samples: 64`
- small `max_new_tokens` for timing validation

**Step 2: Update the issue tracker doc**

Replace stale `50/50` wording with `64/64`, record the two known root causes, and state explicitly that probe runs do not persist checkpoint files.

### Task 4: Run Communication-Only Probe64 Train And In-Memory Eval

**Files:**
- No required repo edits if earlier tasks are complete

**Step 1: Launch training**

Run:
`CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_probe64_comm_only.yaml --max_samples 64`

**Step 2: Inspect train logs**

Verify:
- loss is finite
- graph loss terms are finite
- gradients are non-zero when expected
- per-GPU batch size is 32 and effective batch size is 64
- GPU memory/utilization is consistent with the high-memory probe intention

**Step 3: Run immediate in-memory eval**

Evaluate the trained live model on `64` train samples and `64` test samples with a reduced `max_new_tokens`, and capture only lightweight result files.

**Step 4: Preserve only required artifacts**

Keep only lightweight artifacts needed for docs and comparison: config copy, `loss_log.csv`, and eval/result JSON. Do not persist checkpoints.

### Task 5: Measure Accuracy And Latency From The In-Memory Eval Path

**Files:**
- No required repo edits if earlier tasks are complete

**Step 1: Evaluate on 64 train samples**

Use the post-training in-memory model state to record:
- train accuracy
- avg train sample latency

**Step 2: Evaluate on 64 test samples**

Use the same live model state on `split=test` and record:
- accuracy
- avg sample latency
- avg generated tokens
- avg tokens per second

**Step 3: Decide whether fallback is required**

If either training does not converge cleanly or accuracy stays below the user target, proceed to Task 6.

### Task 6: Run Full-Finetune Probe64 If Needed

**Files:**
- No required repo edits if earlier tasks are complete

**Step 1: Launch the full-finetune probe**

Run:
`CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml --max_samples 64`

**Step 2: Repeat train/test eval**

Run the same in-memory train/test `64`-sample eval loop and compare metrics against the communication-only probe.

**Step 3: Retain only lightweight evidence**

If the fallback probe supersedes the earlier run, keep only the lightweight logs/results referenced in docs and remove stale run directories that do not add information.

### Task 7: Update Versioned Docs With Results

**Files:**
- Modify: `docs/change_log_2026-03-19.md`
- Modify: `docs/current_version_issues_2026-03-19.md`
- Add or modify: `docs/probe64_experiment_log_2026-03-19.md`

**Step 1: Record each versioned change**

For every code patch and experiment rerun, append:
- what changed
- why it changed
- what command was run
- what the result was

**Step 2: Record any better algorithm ideas without over-implementing**

If a stronger communication mechanism is discovered during debugging, document it as a candidate improvement with rationale and whether it was implemented in this round.

**Step 3: Verify docs are internally consistent**

Check that plan, issue tracker, change log, and experiment log all agree on:
- `64` datapoints
- active configs
- achieved metrics
- retained lightweight artifact paths
