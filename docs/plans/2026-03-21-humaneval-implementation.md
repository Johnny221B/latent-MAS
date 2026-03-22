# HumanEval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add HumanEval dataset support for training and official HumanEval `pass@k` evaluation to the existing Latent-MAS pipeline.

**Architecture:** Reuse the current train/eval CLI structure. Extend the dataset loader with a `humaneval` task that maps HumanEval records into the repository's existing `question/question_id/answer` shape while preserving code-eval metadata. Add a task-specific eval branch in `src/cli/evaluate.py` that writes official sample JSONL, invokes a mockable HumanEval harness wrapper, and stores `pass@k` results in repository-native output files.

**Tech Stack:** Python, PyTorch, Hugging Face `datasets`, JSONL, official `human_eval` harness, pytest

---

### Task 1: Add failing dataset coverage for HumanEval records

**Files:**
- Modify: `tests/test_dataset_ids.py`
- Test: `tests/test_dataset_ids.py`

**Step 1: Write the failing test**

Add a test that constructs a `MultiAgentDataset` instance manually with a HumanEval-like raw record and asserts that:

- `question_id == task_id`
- `question == prompt`
- `answer == canonical_solution`
- `test` and `entry_point` survive in the returned sample

**Step 2: Run test to verify it fails**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_dataset_ids.py`
Expected: FAIL because `MultiAgentDataset.__getitem__` currently drops eval-only metadata.

**Step 3: Write minimal implementation**

Update `data/dataset.py` so task configs can declare extra passthrough fields, and make the dataset return those fields for `humaneval`.

**Step 4: Run test to verify it passes**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_dataset_ids.py`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_dataset_ids.py data/dataset.py
git commit -m "feat: add humaneval dataset mapping"
```

### Task 2: Register the HumanEval task and config defaults

**Files:**
- Modify: `data/dataset.py`
- Add: `configs/experiments/humaneval_5agent.yaml`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Add a config test that loads `configs/experiments/humaneval_5agent.yaml` and asserts:

- `training.task == "humaneval"`
- `evaluation.metric == "pass_at_k"`
- `evaluation.num_samples_per_task` is present
- `evaluation.pass_at_k` is present
- `evaluation.do_sample is True`

**Step 2: Run test to verify it fails**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_config.py`
Expected: FAIL because the config file does not exist yet.

**Step 3: Write minimal implementation**

Add the `humaneval` task config in `data/dataset.py` and create the new experiment config file based on the GSM8K layout, with HumanEval-specific eval keys.

**Step 4: Run test to verify it passes**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_config.py`
Expected: PASS

**Step 5: Commit**

```bash
git add data/dataset.py configs/experiments/humaneval_5agent.yaml tests/test_config.py
git commit -m "feat: add humaneval experiment config"
```

### Task 3: Add failing eval test for official sample JSONL generation

**Files:**
- Modify: `tests/test_evaluate_streaming.py`
- Test: `tests/test_evaluate_streaming.py`

**Step 1: Write the failing test**

Add a test that mocks a `humaneval` dataset and a dummy system, runs HumanEval evaluation in a temp output dir, and asserts that:

- `samples.jsonl` is written
- each row contains `task_id` and `completion`
- exact-match fields like `correct` are not used as the primary metric path

**Step 2: Run test to verify it fails**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_evaluate_streaming.py`
Expected: FAIL because `evaluate_loaded_system()` currently uses exact-match scoring for all tasks.

**Step 3: Write minimal implementation**

Refactor `src/cli/evaluate.py` so task-specific evaluation is routed through separate helper functions, then implement a HumanEval helper that emits official sample JSONL.

**Step 4: Run test to verify it passes**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_evaluate_streaming.py`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_evaluate_streaming.py src/cli/evaluate.py
git commit -m "feat: add humaneval sample generation"
```

### Task 4: Add failing test for `pass@k` result ingestion

**Files:**
- Modify: `tests/test_evaluate_streaming.py`
- Modify: `src/cli/evaluate.py`
- Test: `tests/test_evaluate_streaming.py`

**Step 1: Write the failing test**

Add a test around a mockable harness wrapper that returns a fake official results payload and assert that the repository-native summary JSON contains:

- `metrics.pass@1` or equivalent key structure
- configured `k` values
- path references to `samples.jsonl` and the official results file

**Step 2: Run test to verify it fails**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_evaluate_streaming.py`
Expected: FAIL because no HumanEval harness parsing exists yet.

**Step 3: Write minimal implementation**

Implement a dedicated HumanEval harness wrapper and summary writer in `src/cli/evaluate.py`. Make the wrapper easy to monkeypatch in tests and hard-fail with a clear error if `human_eval` is unavailable.

**Step 4: Run test to verify it passes**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_evaluate_streaming.py`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_evaluate_streaming.py src/cli/evaluate.py
git commit -m "feat: add humaneval pass-at-k reporting"
```

### Task 5: Verify train/eval plumbing stays valid for both task families

**Files:**
- Modify: `src/cli/train.py`
- Modify: `src/cli/evaluate.py`
- Test: `tests/test_config.py`
- Test: `tests/test_evaluate_streaming.py`

**Step 1: Write the failing test**

Add or extend tests to assert:

- training can still load task `gsm8k`
- evaluation still accepts existing `gsm8k` config behavior
- HumanEval-specific options are read from config without disturbing existing tasks

**Step 2: Run test to verify it fails**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_config.py tests/test_evaluate_streaming.py`
Expected: FAIL if config/eval branching is incomplete or task assumptions leak across paths.

**Step 3: Write minimal implementation**

Adjust config plumbing and evaluation dispatch so each task family uses the correct metric path while preserving current GSM8K behavior.

**Step 4: Run test to verify it passes**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_config.py tests/test_evaluate_streaming.py`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cli/train.py src/cli/evaluate.py tests/test_config.py tests/test_evaluate_streaming.py
git commit -m "fix: separate humaneval and exact-match eval flows"
```

### Task 6: Update runtime docs to reflect the new HumanEval pipeline

**Files:**
- Modify: `docs/training_pipeline.md`
- Modify: `docs/method.md`
- Modify: `docs/agent_workflow.md`
- Modify: `docs/prompt_flow.md`
- Modify: `docs/ours_json_log_format.md`
- Add or Modify: `docs/README.md`

**Step 1: Write the failing test**

No automated doc test exists. Instead, create a manual doc checklist covering:

- HumanEval dataset support is documented
- official `pass@k` eval flow is documented
- output artifact names are documented
- security and harness prerequisites are documented

**Step 2: Run validation to verify docs are stale before edit**

Run: `rg -n "humaneval|pass@k|samples.jsonl|functional correctness" docs`
Expected: little or no coverage of the new HumanEval pipeline.

**Step 3: Write minimal implementation**

Update the main runtime docs so they describe both answer-matching tasks and HumanEval code-eval behavior. Add `docs/README.md` if needed to satisfy repository documentation policy.

**Step 4: Run validation to verify docs now reflect the implementation**

Run: `rg -n "humaneval|pass@k|samples.jsonl|functional correctness" docs`
Expected: matches in the main docs listed above.

**Step 5: Commit**

```bash
git add docs/training_pipeline.md docs/method.md docs/agent_workflow.md docs/prompt_flow.md docs/ours_json_log_format.md docs/README.md
git commit -m "docs: document humaneval training and eval pipeline"
```

### Task 7: Run final verification before completion

**Files:**
- Verify: `data/dataset.py`
- Verify: `src/cli/train.py`
- Verify: `src/cli/evaluate.py`
- Verify: `tests/test_dataset_ids.py`
- Verify: `tests/test_config.py`
- Verify: `tests/test_evaluate_streaming.py`

**Step 1: Run targeted tests**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_dataset_ids.py tests/test_config.py tests/test_evaluate_streaming.py`
Expected: PASS

**Step 2: Run syntax validation**

Run: `python -m py_compile data/dataset.py src/cli/train.py src/cli/evaluate.py`
Expected: PASS with no output.

**Step 3: Inspect git diff**

Run: `git diff -- data/dataset.py configs/experiments/humaneval_5agent.yaml src/cli/train.py src/cli/evaluate.py tests/test_dataset_ids.py tests/test_config.py tests/test_evaluate_streaming.py docs/training_pipeline.md docs/method.md docs/agent_workflow.md docs/prompt_flow.md docs/ours_json_log_format.md docs/README.md`
Expected: only intended HumanEval-related changes.

**Step 4: Commit**

```bash
git add data/dataset.py configs/experiments/humaneval_5agent.yaml src/cli/train.py src/cli/evaluate.py tests/test_dataset_ids.py tests/test_config.py tests/test_evaluate_streaming.py docs/training_pipeline.md docs/method.md docs/agent_workflow.md docs/prompt_flow.md docs/ours_json_log_format.md docs/README.md
git commit -m "feat: add humaneval training and official eval support"
```
