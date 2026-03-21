# Chat With Text Inference Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an inference-only `chat_with_text` mode that keeps the multi-agent graph but replaces latent-prefix communication with text-message communication for ablation studies.

**Architecture:** Keep training unchanged. During inference, non-terminal agents generate text messages instead of latent trajectories; the DAG executor forwards aggregated upstream text to downstream agents and the terminal agent answers from ordinary chat context without latent prefixes. Preserve adjacency as the graph structure, but do not use compressor outputs in this mode.

**Tech Stack:** Python, PyTorch, Hugging Face generation, pytest, markdown docs.

---

### Task 1: Lock the new mode with failing tests

**Files:**
- Modify: `tests/test_agent_generation.py`
- Modify: `tests/test_evaluate_streaming.py`

**Step 1: Write the failing tests**

- Add a focused agent-level test that expects a new `chat_with_text` inference mode to build chat input without latent prefixes.
- Add an eval-level test that expects CLI/config plumbing to accept `chat_with_text`.

**Step 2: Run tests to verify they fail**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_agent_generation.py tests/test_evaluate_streaming.py`
Expected: FAIL because `chat_with_text` is not implemented.

### Task 2: Implement text-message inference path

**Files:**
- Modify: `src/models/agent.py`
- Modify: `src/graph/dag_executor.py`
- Modify: `src/pipeline/multi_agent_system.py`
- Modify: `src/cli/evaluate.py`

**Step 1: Write minimal implementation**

- Add helper(s) in `Agent` to build chat prompts with upstream text messages and to generate intermediate text messages.
- Extend the DAG executor with a `chat_with_text` inference branch that stores per-agent text outputs instead of compressed prefixes.
- Keep training paths unchanged.
- Extend CLI/config acceptance for `chat_with_text`.

**Step 2: Run tests to verify they pass**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_agent_generation.py tests/test_evaluate_streaming.py`
Expected: PASS

### Task 3: Update runtime docs

**Files:**
- Modify: `docs/training_pipeline.md`
- Modify: `docs/prompt_flow.md`
- Modify: `docs/ours_json_log_format.md`
- Modify: `docs/README.md` if classification changes

**Step 1: Update docs**

- Document that `chat_with_text` is an inference-only ablation mode.
- Explain that adjacency still controls topology, while latent compressor/prefix communication is bypassed.
- Update eval JSON field descriptions if new metadata appears in generation or agent logs.

**Step 2: Verify docs and code references stay aligned**

Run: `uv run --python .venv/bin/python -m pytest -q tests/test_agent_generation.py tests/test_evaluate_streaming.py`
Expected: PASS

