# Resume Training From Checkpoint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a config-driven training resume path so `scripts/train.sh` can restart from an intermediate checkpoint and continue writing outputs into the same run directory.

**Architecture:** Extend config normalization with a new `training.resume_from_checkpoint` key, add train-time helpers to resolve resume metadata and load checkpoint state, and reuse the checkpoint parent directory instead of creating a new timestamped output directory when resuming. Keep the non-resume path unchanged.

**Tech Stack:** Python, PyTorch, pytest, YAML configs, project docs

---

### Task 1: Add failing tests for config normalization and resume path resolution

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_train.py`

**Step 1: Write the failing test**

```python
def test_load_config_defaults_resume_from_checkpoint_to_none(tmp_path: Path):
    ...
    assert loaded["training"]["resume_from_checkpoint"] is None


def test_resolve_resume_checkpoint_returns_none_without_config():
    ...
    assert resolve_resume_checkpoint({}) is None


def test_resolve_output_dir_uses_resume_checkpoint_parent(tmp_path: Path):
    ...
    assert resolve_training_output_dir(config) == checkpoint_path.parent
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py tests/test_train.py -k "resume_from_checkpoint or resolve_resume or resolve_training_output_dir" -v`

Expected: FAIL because helper(s) and normalization do not exist yet.

**Step 3: Write minimal implementation**

Add normalization default and introduce train helper function stubs with the expected behavior.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py tests/test_train.py -k "resume_from_checkpoint or resolve_resume or resolve_training_output_dir" -v`

Expected: PASS

### Task 2: Add failing tests for checkpoint loading semantics

**Files:**
- Modify: `tests/test_train.py`

**Step 1: Write the failing test**

```python
def test_load_training_checkpoint_restores_trainable_states(tmp_path: Path):
    ...
    state = load_training_checkpoint(...)
    assert state["global_step"] == 3000
    assert optimizer.state_dict()["state"] == expected_state
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -k "load_training_checkpoint" -v`

Expected: FAIL because the loader helper does not exist yet.

**Step 3: Write minimal implementation**

Add a checkpoint loader helper that restores compressor, adjacency, optimizer, and optional base-model state.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -k "load_training_checkpoint" -v`

Expected: PASS

### Task 3: Wire resume support into the train entrypoint

**Files:**
- Modify: `src/cli/train.py`
- Modify: `src/utils/config.py`

**Step 1: Update output-dir selection**

When `training.resume_from_checkpoint` is set, resolve the checkpoint path early and reuse `checkpoint.parent` as `output_dir`.

**Step 2: Update checkpoint restore flow**

After constructing the system and optimizer, restore checkpoint state before entering the training loop and seed `global_step` from the checkpoint.

**Step 3: Preserve existing behavior**

If `training.resume_from_checkpoint` is absent or null, continue creating a timestamped output directory and starting from step 0.

**Step 4: Run targeted tests**

Run: `pytest tests/test_config.py tests/test_train.py -k "resume" -v`

Expected: PASS

### Task 4: Update experiment config and docs

**Files:**
- Modify: `configs/experiments/am_deepseek_r1_distilled_5agent.yaml`
- Modify: `docs/training_pipeline.md`

**Step 1: Add config key to the AM experiment**

Add `training.resume_from_checkpoint: null` so the resume interface is visible in the main config.

**Step 2: Document runtime behavior**

Describe that resuming from `checkpoint_step*.pt` restores training state and forces outputs to continue in the original run directory.

**Step 3: Run documentation-adjacent tests if any**

Run: `pytest tests/test_config.py::test_am_deepseek_r1_distilled_experiment_config_matches_competition_math_shape -v`

Expected: PASS

### Task 5: Verify the full change set

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_train.py`
- Modify: `src/utils/config.py`
- Modify: `src/cli/train.py`
- Modify: `configs/experiments/am_deepseek_r1_distilled_5agent.yaml`
- Modify: `docs/training_pipeline.md`

**Step 1: Run the focused verification suite**

Run: `pytest tests/test_config.py tests/test_train.py -v`

Expected: PASS

**Step 2: Spot-check the target checkpoint path shape**

Run: `ls -la outputs/am_deepseek_r1_distilled_qwen3-4b_20260326_130242`

Expected: directory contains `checkpoint_step3000.pt`.

**Step 3: Summarize exact resume usage**

Document the config snippet the user needs and note that resumed artifacts stay in the checkpoint folder.
