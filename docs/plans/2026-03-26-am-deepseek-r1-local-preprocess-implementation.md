# AM DeepSeek R1 Local Preprocess Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `am_deepseek_r1_distilled` training read from a prepared local `jsonl` file so `bash scripts/train.sh` can run on `c1103a-s5` without online dataset loading failures.

**Architecture:** Introduce a dedicated preparation script that downloads and normalizes the Hugging Face subsets into `data/am_deepseek_r1_distilled/train.jsonl`. Change the task loader so it reads only this local file, raises a clear missing-file error otherwise, and keeps the same normalized sample contract already used by training.

**Tech Stack:** Python, Hugging Face `datasets`, JSONL, pytest, remote SSH execution

---

### Task 1: Lock the new local-file contract with failing tests

**Files:**
- Modify: `tests/test_dataset_ids.py`
- Create: `tests/test_prepare_am_deepseek_r1_distilled.py`
- Test: `tests/test_dataset_ids.py`
- Test: `tests/test_prepare_am_deepseek_r1_distilled.py`

**Step 1: Write the failing tests**

Add tests that:

- verify the loader reads `data/am_deepseek_r1_distilled/train.jsonl` instead of Hugging Face
- verify the loaded sample preserves full assistant output with `<think>` and `<answer>`
- verify missing local file raises a clear error mentioning the prepare script
- verify the prepare-script normalization turns raw subset examples into the expected JSONL rows

**Step 2: Run tests to verify they fail**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q \
  tests/test_dataset_ids.py \
  tests/test_prepare_am_deepseek_r1_distilled.py
```

Expected: FAIL because the loader still reads Hugging Face directly and the prepare script does not exist yet.

### Task 2: Implement the prepare script

**Files:**
- Create: `scripts/prepare_am_deepseek_r1_distilled.py`
- Test: `tests/test_prepare_am_deepseek_r1_distilled.py`

**Step 1: Write minimal implementation**

Implement:

- raw subset loader
- message extraction helpers
- deterministic `question_id` generation
- row normalization
- JSONL writer to `data/am_deepseek_r1_distilled/train.jsonl`
- CLI entrypoint with optional `--output`

**Step 2: Run script tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q \
  tests/test_prepare_am_deepseek_r1_distilled.py
```

Expected: PASS

### Task 3: Switch runtime loader to local prepared data

**Files:**
- Modify: `src/data/am_deepseek_r1_distilled.py`
- Modify: `tests/test_dataset_ids.py`
- Test: `tests/test_dataset_ids.py`

**Step 1: Write minimal implementation**

Change the loader so it:

- reads local `jsonl`
- supports only `train`
- errors clearly if the file is absent
- keeps the same normalized field mapping for `MultiAgentDataset`

**Step 2: Run loader tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q \
  tests/test_dataset_ids.py \
  tests/test_prepare_am_deepseek_r1_distilled.py
```

Expected: PASS

### Task 4: Update docs to match the new runtime contract

**Files:**
- Modify: `docs/data/am_deepseek_r1_distilled.md`
- Modify: `docs/training_pipeline.md`

**Step 1: Update docs**

Document:

- training now reads local prepared JSONL
- the prepare script path
- the required runtime sequence: prepare first, train second

**Step 2: Review docs against code**

Check the file path and error wording match the implementation exactly.

### Task 5: Verify locally and run remotely

**Files:**
- Verify: `scripts/prepare_am_deepseek_r1_distilled.py`
- Verify: `src/data/am_deepseek_r1_distilled.py`
- Verify: `scripts/train.sh`

**Step 1: Local verification**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --python .venv/bin/python -m pytest -q \
  tests/test_dataset_ids.py \
  tests/test_prepare_am_deepseek_r1_distilled.py \
  tests/test_config.py
```

Expected: PASS

**Step 2: Remote data preparation**

Run on `c1103a-s5`:

```bash
cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS
.venv/bin/python scripts/prepare_am_deepseek_r1_distilled.py
```

Expected: `data/am_deepseek_r1_distilled/train.jsonl` is created

**Step 3: Remote training**

Run on `c1103a-s5`:

```bash
cd /blue/buyuheng/chengzhi.ucsb/code/toby/latent-MAS
bash scripts/train.sh
```

Expected: training gets past dataset loading and begins the train loop
