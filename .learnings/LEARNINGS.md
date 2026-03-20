## [LRN-20260319-001] correction

**Logged**: 2026-03-19T00:00:00Z
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
Execution should remain iterative until the user-defined metric target is reached, not stop after a single planned phase.

### Details
The user clarified that the earlier phase-based plan should be treated as an iterative loop with continued optimization toward the stated target, and that the current program's known issues also need a dedicated documentation artifact. Future implementation should therefore treat phase boundaries as checkpoints rather than stopping conditions.

### Suggested Action
Keep iterating on training/inference changes and experiments until the target is met or a hard blocker is proven, and add a separate docs file that tracks current-version issues, fixes, and remaining gaps.

### Metadata
- Source: user_feedback
- Related Files: docs
- Tags: iteration, planning, docs

---

## [LRN-20260319-002] best_practice

**Logged**: 2026-03-19T00:00:00Z
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
Metrics-schema helpers should grow compatibly by adding optional fields with defaults first.

### Details
During iterative logging improvements, a helper function signature was extended with required positional parameters, which immediately broke unrelated tests. This kind of helper is shared by tests and runtime code, so additive changes should default to `None` and keep old callers working.

### Suggested Action
Prefer backwards-compatible defaults whenever expanding structured logging helpers, then migrate call sites incrementally.

### Metadata
- Source: error
- Related Files: src/cli/evaluate.py
- Tags: compatibility, logging, tests

---

## [LRN-20260319-003] correction

**Logged**: 2026-03-19T00:00:00Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
`eval` must remain on `chat_with_prefix`; consistency should be restored by aligning training to chat mode rather than falling back to `legacy_plain_with_prefix`.

### Details
While debugging low evaluation accuracy, switching evaluation to `legacy_plain_with_prefix` looked like a useful diagnostic, but the user explicitly requires the production inference path to stay on `chat_with_prefix`. The correct direction is therefore to make training and chat-style inference consistent, not to optimize around a legacy path.

### Suggested Action
Keep `chat_with_prefix` as the only accepted evaluation path for the target metric, and update training/prompt construction so the terminal path is trained on the same chat-style input format used at inference time.

### Metadata
- Source: user_feedback
- Related Files: src/models/agent.py, src/cli/evaluate.py
- Tags: chat_with_prefix, train_eval_alignment

---

## [LRN-20260319-004] correction

**Logged**: 2026-03-19T00:00:00Z
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
Large experimental artifacts must be cleaned up immediately once they are no longer needed.

### Details
During iterative probe training, I left multi-GB checkpoints and model files in `outputs/` longer than necessary, which contributed to the workspace filesystem filling up. The user explicitly corrected this workflow. For future runs, any superseded checkpoints, corrupted saves, or obsolete probe artifacts should be removed in the same session as soon as their metrics/logs have been preserved.

### Suggested Action
After each experiment milestone, keep only the files still needed for the next step or for final reporting. Delete stale checkpoints promptly, especially when working with 8B-scale DDP runs.

### Metadata
- Source: user_feedback
- Related Files: outputs, docs/change_log_2026-03-19.md
- Tags: disk, cleanup, experiment_artifacts

---

## [LRN-20260319-005] correction

**Logged**: 2026-03-19T19:15:00Z
**Priority**: high
**Status**: pending
**Area**: config

### Summary
The active small-probe plan must target `64` datapoints, not `50`.

### Details
While proposing the next repair-and-rerun plan, I framed the validation loop around `50` train/test samples because that matched an earlier request. The user corrected the active requirement: this round must use `64` datapoints, and the plan should be treated as a living document that keeps being updated during execution.

### Suggested Action
Standardize the current rerun plan, configs, validation commands, and docs updates around `64` samples and keep the plan document synchronized with execution results.

### Metadata
- Source: user_feedback
- Related Files: configs/experiments/gsm8k_5agent_probe64_comm_only.yaml, configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml, docs/plans
- Tags: datapoints, planning, probe64

---

## [LRN-20260319-006] correction

**Logged**: 2026-03-19T19:19:00Z
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
The active probe workflow must not persist checkpoint files at all.

### Details
The user tightened the artifact constraint further: this round should not store checkpoint files because they are too large. Keeping only a "winning checkpoint" is still too heavy for the current workspace constraint. The correct direction is to run evaluation directly from the trained in-memory model state, store only lightweight CSV/JSON summaries, and avoid `torch.save(...)` model artifacts for probe runs.

### Suggested Action
Change the probe64 execution path so training can hand the live model state directly into evaluation, and disable checkpoint emission entirely for these runs.

### Metadata
- Source: user_feedback
- Related Files: src/cli/train.py, src/cli/evaluate.py, docs/plans/2026-03-19-probe64-train-eval-repair.md
- Tags: checkpoints, disk, probe64, train_eval_loop

---

## [LRN-20260319-007] best_practice

**Logged**: 2026-03-19T20:35:00Z
**Priority**: high
**Status**: pending
**Area**: config

### Summary
`max_new_tokens=64` is too low for the probe64 GSM8K eval loop and can hide model capability behind systematic truncation.

### Details
On the communication-only probe, every evaluated sample stopped exactly at `generated_token_count=64`, and several generations contained the correct reasoning trajectory but were cut before the final answer marker. This made the reported accuracy artificially worse than the underlying generation quality. Raising the eval cap to `128` is still lightweight but avoids that specific truncation failure mode.

### Suggested Action
Use at least `128` eval tokens for the current probe64 GSM8K runs unless there is direct evidence that a lower cap preserves final-answer coverage.

### Metadata
- Source: conversation
- Related Files: configs/experiments/gsm8k_5agent_probe64_comm_only.yaml, configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml
- Tags: evaluation, truncation, gsm8k

---

## [LRN-20260319-008] best_practice

**Logged**: 2026-03-19T20:36:00Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
On 64-sample GSM8K probes, same-split accuracy and held-out accuracy must be reported separately because aggressive full-finetuning can hit 100% train accuracy while collapsing on test.

### Details
The `1e-4 x 64 epochs` full-finetune probe reached `100%` accuracy on the 64 training samples while dropping to `3.12%` on 64 held-out test samples, often emitting memorized short answers from the training set. This means "accuracy > 80%" is ambiguous unless the evaluation split is named explicitly. Same-split probe success does not imply generalization.

### Suggested Action
Whenever running tiny-data probe experiments, always record both train-split and test-split metrics and explicitly distinguish memorization from generalization in docs and summaries.

### Metadata
- Source: conversation
- Related Files: docs/probe64_experiment_log_2026-03-19.md, outputs/gsm8k_qwen3-8b_probe64_full_finetune_20260319_152513
- Tags: overfitting, evaluation, probe64

---
