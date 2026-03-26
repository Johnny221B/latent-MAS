# AM DeepSeek R1 Distilled Dataset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the `am_deepseek_r1_distilled` dataset task so training can supervise on full assistant outputs, including `<think>` and `<answer>` tags, from the merged `am_0.5M` and `am_0.9M` subsets.

**Architecture:** Add a dedicated dataset module under `src/data/` that loads both Hugging Face subsets, flattens the nested metadata needed by the repository, extracts `user` and `assistant` messages into the normalized `question` and `answer` fields, and generates a deterministic hash-based `question_id`. Register the new task in the central factory, cover the behavior with dataset-level tests first, then update runtime docs to keep the dataset contract aligned with actual code behavior.

**Tech Stack:** Python, Hugging Face `datasets`, pytest, repository docs in Markdown

---

### Task 1: Add failing coverage for the new dataset contract

**Files:**
- Modify: `tests/test_dataset_ids.py`
- Test: `tests/test_dataset_ids.py`

**Step 1: Write the failing test**

Add tests that:

- assert `am_deepseek_r1_distilled` appears in `get_task_configs()`
- mock both subset loads for `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`
- assert dataset length is the sum of both subsets
- assert `question` comes from the `user` message
- assert `answer` equals the full `assistant` message with `<think>` and `<answer>` tags
- assert metadata fields are preserved
- assert `question_id` is deterministic and starts with `am-r1-`

**Step 2: Run test to verify it fails**

Run:

```bash
uv run --python .venv/bin/python -m pytest -q tests/test_dataset_ids.py
```

Expected: FAIL because the new task is not registered and no dataset module exists yet.

**Step 3: Write minimal implementation**

Do not implement yet. This task ends at a verified failing test.

**Step 4: Commit**

```bash
git add tests/test_dataset_ids.py
git commit -m "test: add AM DeepSeek R1 distilled dataset coverage"
```

### Task 2: Implement the dataset module and register the task

**Files:**
- Create: `src/data/am_deepseek_r1_distilled.py`
- Modify: `src/data/factory.py`
- Test: `tests/test_dataset_ids.py`

**Step 1: Write minimal implementation**

Implement:

- `_load_hf_dataset(dataset_name, subset, split)`
- `_load_concat_train_split()` to load `am_0.5M` and `am_0.9M`, tag rows with `subset`, flatten `info` into top-level metadata fields, and concatenate the two datasets
- helper(s) to extract the `user` and `assistant` message contents
- helper to generate deterministic `am-r1-<sha1-prefix>` ids
- `build_task_configs()` returning the new task config
- task registration in `src/data/factory.py`

Use `question_formatter`, `answer_extractor`, and `extra_fields` only where they match the existing data abstraction. Keep the implementation minimal and task-local.

**Step 2: Run test to verify it passes**

Run:

```bash
uv run --python .venv/bin/python -m pytest -q tests/test_dataset_ids.py
```

Expected: PASS for the new dataset coverage and existing dataset ID tests.

**Step 3: Refactor if needed**

Only if necessary:

- keep helper names explicit
- keep field flattening local to the new module
- avoid touching unrelated datasets

**Step 4: Commit**

```bash
git add src/data/am_deepseek_r1_distilled.py src/data/factory.py tests/test_dataset_ids.py
git commit -m "feat(data): add AM DeepSeek R1 distilled dataset"
```

### Task 3: Document the new dataset contract

**Files:**
- Create: `docs/data/am_deepseek_r1_distilled.md`
- Modify: `docs/data/README.md`
- Modify: `docs/training_pipeline.md`
- Modify: `docs/README.md`

**Step 1: Update docs**

Document:

- source dataset and subset merge behavior
- `train`-only split semantics
- `question` and `answer` mapping from `messages`
- full assistant supervision including `<think>` and `<answer>`
- preserved metadata fields

Also add the new dataset doc to the docs indexes.

**Step 2: Verify docs reflect runtime behavior**

Review the new module and ensure the docs match the implemented fields and split semantics exactly.

**Step 3: Commit**

```bash
git add docs/data/am_deepseek_r1_distilled.md docs/data/README.md docs/training_pipeline.md docs/README.md
git commit -m "docs: add AM DeepSeek R1 distilled dataset docs"
```

### Task 4: Final verification

**Files:**
- Verify: `src/data/am_deepseek_r1_distilled.py`
- Verify: `src/data/factory.py`
- Verify: `tests/test_dataset_ids.py`
- Verify: `docs/data/am_deepseek_r1_distilled.md`
- Verify: `docs/data/README.md`
- Verify: `docs/training_pipeline.md`
- Verify: `docs/README.md`

**Step 1: Run targeted tests**

Run:

```bash
uv run --python .venv/bin/python -m pytest -q tests/test_dataset_ids.py
```

Expected: PASS

**Step 2: Run syntax verification**

Run:

```bash
python -m py_compile src/data/am_deepseek_r1_distilled.py src/data/factory.py
```

Expected: PASS with no output

**Step 3: Review diff**

Run:

```bash
git diff -- src/data/am_deepseek_r1_distilled.py src/data/factory.py tests/test_dataset_ids.py docs/data/am_deepseek_r1_distilled.md docs/data/README.md docs/training_pipeline.md docs/README.md
```

Expected: only the planned dataset, test, and doc changes

**Step 4: Commit**

```bash
git add src/data/am_deepseek_r1_distilled.py src/data/factory.py tests/test_dataset_ids.py docs/data/am_deepseek_r1_distilled.md docs/data/README.md docs/training_pipeline.md docs/README.md
git commit -m "feat: add AM DeepSeek R1 distilled training dataset"
```
