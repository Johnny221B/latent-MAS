# AM DeepSeek R1 Source Filter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add source-based filtering for `am_deepseek_r1_distilled` by rebuilding a `question_id -> source` index from Hugging Face raw rows while keeping `train.jsonl` unchanged.

**Architecture:** Extend the prepare script to generate a separate `source_index.jsonl` artifact from the original HF subsets. Thread optional task-specific dataset kwargs from config into dataset creation, and make the AM DeepSeek loader filter local `train.jsonl` rows by matching `question_id` values against selected `training.sources`.

**Tech Stack:** Python, local JSONL files, pytest, YAML config loading

---

### Task 1: Lock source-index preparation with failing tests

**Files:**
- Modify: `tests/test_prepare_am_deepseek_r1_distilled.py`
- Test: `tests/test_prepare_am_deepseek_r1_distilled.py`

**Step 1: Write the failing test**

Add tests that:

- verify the prepare module writes `source_index.jsonl`
- verify each index row contains `question_id` and `source`
- verify `question_id` matches the same deterministic function used for `train.jsonl`
- verify source extraction comes from raw sample metadata rather than normalized train rows

**Step 2: Run test to verify it fails**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q tests/test_prepare_am_deepseek_r1_distilled.py
```

Expected: FAIL because the prepare script does not yet emit `source_index.jsonl`.

**Step 3: Write minimal implementation**

Do not implement yet. This task ends at a verified failing test.

### Task 2: Lock source-filtered loader behavior with failing tests

**Files:**
- Modify: `tests/test_dataset_ids.py`
- Modify: `tests/test_config.py`
- Test: `tests/test_dataset_ids.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing tests**

Add tests that:

- verify `training.sources` survives config loading
- verify AM DeepSeek loader returns all rows when no sources are supplied
- verify loader keeps only matching rows when `sources=["..."]`
- verify loader errors when `sources` is set but `source_index.jsonl` is missing
- verify loader errors when requested sources are not present in the index

**Step 2: Run test to verify it fails**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q tests/test_dataset_ids.py tests/test_config.py
```

Expected: FAIL because dataset creation does not yet accept task-specific source filters.

**Step 3: Write minimal implementation**

Do not implement yet. This task ends at a verified failing test.

### Task 3: Emit `source_index.jsonl` during preparation

**Files:**
- Modify: `scripts/prepare_am_deepseek_r1_distilled.py`
- Test: `tests/test_prepare_am_deepseek_r1_distilled.py`

**Step 1: Write minimal implementation**

Implement:

- helper to extract source from the raw HF sample
- writer for `source_index.jsonl`
- stable reuse of the existing `question_id` function
- CLI behavior that writes both `train.jsonl` and `source_index.jsonl` in one prepare run, without changing train-row schema

**Step 2: Run targeted tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q tests/test_prepare_am_deepseek_r1_distilled.py
```

Expected: PASS

### Task 4: Thread dataset kwargs and implement source filtering

**Files:**
- Modify: `src/data/base.py`
- Modify: `src/data/factory.py`
- Modify: `src/cli/train.py`
- Modify: `src/data/am_deepseek_r1_distilled.py`
- Test: `tests/test_dataset_ids.py`
- Test: `tests/test_config.py`

**Step 1: Write minimal implementation**

Implement:

- optional `dataset_kwargs` plumbing from training config to `create_dataset(...)`
- AM DeepSeek loader support for `sources`
- index reader for `source_index.jsonl`
- filtering of the existing local JSONL dataset by `question_id`
- clear errors for missing index and unknown sources

Keep the change task-local: other datasets should ignore extra kwargs they do not use.

**Step 2: Run targeted tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q tests/test_dataset_ids.py tests/test_config.py
```

Expected: PASS

### Task 5: Update docs for source-index workflow

**Files:**
- Modify: `docs/data/am_deepseek_r1_distilled.md`
- Modify: `docs/training_pipeline.md`
- Optionally Modify: `docs/README.md`
- Optionally Modify: `docs/data/README.md`

**Step 1: Document runtime behavior**

Document:

- `train.jsonl` remains unchanged
- `source_index.jsonl` is regenerated from HF raw rows
- `training.sources` filters samples by joining on `question_id`
- missing index / unknown source failures

**Step 2: Verify docs reflect runtime**

Re-read the changed code paths and ensure the docs match the implemented behavior exactly.

### Task 6: Full verification

**Files:**
- Verify: `scripts/prepare_am_deepseek_r1_distilled.py`
- Verify: `src/data/am_deepseek_r1_distilled.py`
- Verify: `src/data/base.py`
- Verify: `src/data/factory.py`
- Verify: `src/cli/train.py`
- Verify: `tests/test_prepare_am_deepseek_r1_distilled.py`
- Verify: `tests/test_dataset_ids.py`
- Verify: `tests/test_config.py`

**Step 1: Run verification suite**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q \
  tests/test_prepare_am_deepseek_r1_distilled.py \
  tests/test_dataset_ids.py \
  tests/test_config.py
```

Expected: PASS

**Step 2: Run syntax verification**

Run:

```bash
.venv/bin/python -m py_compile \
  scripts/prepare_am_deepseek_r1_distilled.py \
  src/data/am_deepseek_r1_distilled.py \
  src/data/base.py \
  src/data/factory.py \
  src/cli/train.py
```

Expected: no output, exit code 0
