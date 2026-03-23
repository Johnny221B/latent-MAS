# Competition Math Solution Supervision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change `competition_math` training supervision from extracted final-answer strings to full `solution` text while keeping evaluation/probe answer extraction unchanged.

**Architecture:** Update the dataset mapping so `MultiAgentDataset(task="competition_math")` returns raw `solution` as `answer`. Keep `src/utils/answer_extraction.py` unchanged so inference-time scoring still extracts the final boxed answer or trailing answer span from generated text.

**Tech Stack:** Python, PyTorch dataset wrappers, pytest, Markdown docs

---

### Task 1: Lock the new dataset contract with a failing test

**Files:**
- Modify: `tests/test_dataset_ids.py`
- Test: `tests/test_dataset_ids.py`

**Step 1: Write the failing test**

Update the competition-math dataset test so it expects:

```python
assert sample["answer"] == "We get 42, so the final answer is \\boxed{42}."
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_ids.py::test_competition_math_dataset_uses_full_solution_as_training_target -q`

Expected: FAIL because the dataset still returns `"42"`.

**Step 3: Write minimal implementation**

Remove the dataset-level `answer_extractor` from `src/data/competition_math.py` so `answer_field="solution"` flows through unchanged.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_ids.py::test_competition_math_dataset_uses_full_solution_as_training_target -q`

Expected: PASS

### Task 2: Update documentation to match runtime behavior

**Files:**
- Modify: `docs/training_pipeline.md`
- Modify: `docs/method.md`
- Modify: `docs/data/competition_math.md`

**Step 1: Document the new supervision target**

State that `competition_math` now trains on full `solution` text, not an extracted final answer string.

**Step 2: Preserve eval semantics in docs**

Clarify that evaluation and training-probe scoring still extract the final answer from generated text for exact-match comparison.

**Step 3: Reconcile probe wording**

Ensure `competition_math` probe documentation matches the current config split between the formal config (`samples: 0`) and debug config (`samples: 100`).

### Task 3: Verify the change end-to-end

**Files:**
- Test: `tests/test_dataset_ids.py`
- Test: `tests/test_baseline_scripts.py`

**Step 1: Run focused dataset and extraction tests**

Run:

```bash
pytest tests/test_dataset_ids.py::test_competition_math_dataset_uses_full_solution_as_training_target \
       tests/test_baseline_scripts.py::test_extract_answer_competition_math_prefers_boxed_value -q
```

Expected: PASS

**Step 2: Run a broader regression slice**

Run:

```bash
pytest tests/test_dataset_ids.py tests/test_config.py -q
```

Expected: PASS
