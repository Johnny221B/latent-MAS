## [ERR-20260319-001] pytest

**Logged**: 2026-03-19T00:00:00Z
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
Extending `write_eval_snapshot()` with new timing metrics broke existing tests because the new arguments were added as required positional parameters.

### Error
```text
TypeError: write_eval_snapshot() missing 3 required positional arguments: 'avg_sample_seconds', 'avg_generated_tokens', and 'avg_tokens_per_second'
```

### Context
- Command: `pytest tests/test_dag_executor.py tests/test_agent_generation.py tests/test_evaluate_streaming.py -q`
- Change introduced during iterative evaluation logging updates.

### Suggested Fix
When extending serialized metrics helpers, add backwards-compatible default values unless every caller is updated in the same patch.

### Metadata
- Reproducible: yes
- Related Files: src/cli/evaluate.py

---

## [ERR-20260319-002] dependency_probe

**Logged**: 2026-03-19T00:00:00Z
**Priority**: low
**Status**: pending
**Area**: infra

### Summary
A quick dependency probe failed because the command used `importlib.util` without importing the submodule correctly in the inline script.

### Error
```text
AttributeError: module 'importlib' has no attribute 'util'
```

### Context
- Command: `uv run --python .venv/bin/python python - <<'PY' ...`
- Goal: check whether `peft` and `bitsandbytes` are available before choosing the next training adaptation path.

### Suggested Fix
Use `import importlib.util` or `from importlib import util` in one-off dependency probes.

### Metadata
- Reproducible: yes
- Related Files: .venv

---

## [ERR-20260319-003] probe64_comm_only_bf16_compressor_dtype

**Logged**: 2026-03-19T18:11:37Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
`probe64` communication-only training failed on the first forward pass because the bf16 backbone emitted bf16 hidden states into a float32 compressor.

### Error
```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float
```

### Context
- Command: `CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_probe64_comm_only.yaml --max_samples 64`
- Failure site: `src/models/compressor.py` inside `nn.MultiheadAttention`
- Trigger: enabling `model.dtype: bfloat16` for probe configs while leaving compressor parameters in float32

### Suggested Fix
Align compressor inputs to the compressor parameter dtype (or explicitly autocast the compressor) before cross-attention, then rerun the probe.

### Metadata
- Reproducible: yes
- Related Files: src/models/compressor.py, configs/experiments/gsm8k_5agent_probe64_comm_only.yaml
- See Also: ERR-20260319-002

---

## [ERR-20260319-004] probe64_full_ft_ddp_wrapper_access

**Logged**: 2026-03-19T18:18:08Z
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
`full_finetune` training failed because `BaseModelWrapper` still accesses base-model helper methods directly on a DDP wrapper.

### Error
```text
AttributeError: 'DistributedDataParallel' object has no attribute 'get_input_embeddings'
```

### Context
- Command: `CUDA_VISIBLE_DEVICES=0,1 uv run --python .venv/bin/python torchrun --nproc_per_node=2 src/cli/train.py --config configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml --max_samples 64`
- Failure site: `src/models/base_model.py:get_input_embeddings()` during non-terminal latent reasoning
- Trigger: `training.train_strategy=full_finetune` wraps `base_model.model` with DDP, but helper methods still assume a raw HF model

### Suggested Fix
Make `BaseModelWrapper` unwrap DDP transparently for helper access (`get_input_embeddings`, `lm_head`, and similar helper reads), then rerun full-finetune.

### Metadata
- Reproducible: yes
- Related Files: src/models/base_model.py, src/cli/train.py
- See Also: ERR-20260319-003

---

## [ERR-20260319-005] apply_patch_context_drift

**Logged**: 2026-03-19T20:40:00Z
**Priority**: low
**Status**: pending
**Area**: docs

### Summary
A large multi-file `apply_patch` failed because expected context in the docs had drifted from the patch assumptions.

### Error
```text
apply_patch verification failed: Failed to find expected lines in docs/current_version_issues_2026-03-19.md
```

### Context
- Operation attempted: one large patch updating learnings, multiple docs, and adding an experiment log in a single tool call
- The target docs had different surrounding lines than the patch expected

### Suggested Fix
Split large documentation edits into smaller file-specific patches after re-reading the current file contents.

### Metadata
- Reproducible: yes
- Related Files: docs/current_version_issues_2026-03-19.md, docs/change_log_2026-03-19.md

---
## [ERR-20260320-001] using-git-worktrees

**Logged**: 2026-03-20T00:00:00Z
**Priority**: medium
**Status**: pending
**Area**: docs

### Summary
Global git worktree creation succeeded, but subsequent file mutations in that worktree failed under sandbox permissions.

### Error
```text
mkdir: cannot create directory ‘docs/reference’: Read-only file system
mkdir: cannot create directory ‘docs/records’: Read-only file system
```

### Context
- Command/operation attempted: create docs subdirectories and move docs files inside `~/.config/superpowers/worktrees/latent-MAS/docs-train-pipeline-refresh`
- The repository worktree lives outside the writable roots available to this Codex session
- The task requires editing tracked files, so the worktree became unusable for implementation under current sandbox constraints

### Suggested Fix
Prefer project-local or otherwise writable worktree locations for sessions that must edit files, or fall back to the original writable workspace when the chosen global worktree path is sandbox read-only.

### Metadata
- Reproducible: yes
- Related Files: .learnings/ERRORS.md

---

## [ERR-20260320-002] git-checkout

**Logged**: 2026-03-20T00:00:00Z
**Priority**: medium
**Status**: pending
**Area**: config

### Summary
Branch checkout failed in the sandbox because git could not create `index.lock`.

### Error
```text
fatal: Unable to create '/home/chengzhi.ucsb/code/toby/latent-MAS/.git/index.lock': Read-only file system
```

### Context
- Command/operation attempted: `git checkout toby`
- Input or parameters used: switch from `fix/probe-live-eval-fulltrain` to `toby` after committing docs changes
- Environment details if relevant: repository content is writable, but this branch switch required git metadata writes that the current sandbox denied

### Suggested Fix
Rerun the checkout with escalated permissions when branch switching is required for the task.

### Metadata
- Reproducible: yes
- Related Files: .learnings/ERRORS.md

---
