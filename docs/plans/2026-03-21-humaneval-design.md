# HumanEval Train And Eval Design

## Goal

Add `HumanEval` to the existing Latent-MAS train/eval pipeline so that:

- training uses HumanEval code-generation samples
- evaluation uses the official HumanEval functional-correctness harness
- the user-facing workflow stays close to the current `gsm8k` CLI/config flow

## Scope

This design covers:

- dataset integration
- training target shape
- evaluation behavior
- configuration additions
- test coverage
- documentation updates

This design does not include:

- reward-based training
- execution-feedback training
- a general task-plugin refactor across the whole repository

## Current Constraints

The current repository assumes `gsm8k`-style tasks in several places:

- `data/dataset.py` maps each record to `question_id`, `question`, `answer`
- `src/cli/train.py` trains with teacher forcing on answer text
- `src/cli/evaluate.py` evaluates by generating one answer string and comparing it to gold with task-specific extraction
- `src/utils/answer_extraction.py` is built for string answers such as numbers or multiple-choice letters

`HumanEval` does not fit that exact pattern. Official evaluation expects model-generated code completions and computes `pass@k` by running hidden tests, not by string matching.

## Chosen Approach

Use the existing train/eval entrypoints and add a `humaneval` task-specific path inside the current framework.

This is the lowest-risk option because it preserves:

- `src/cli/train.py`
- `src/cli/evaluate.py`
- timestamped output directories
- distributed launch conventions
- most of the config structure

The repository will therefore support at least two eval families:

- answer-matching tasks such as `gsm8k`
- code-execution tasks such as `humaneval`

## Data Design

### Dataset Source

The preferred source is a Hugging Face dataset mirror compatible with OpenAI HumanEval content. The target sample shape is the standard HumanEval record:

- `task_id`
- `prompt`
- `canonical_solution`
- `test`
- `entry_point`

If needed later, the loader can be extended to support local JSONL fallback, but that is not required for the first implementation.

### Unified Sample Shape

To preserve compatibility with the current training code, dataset items should still expose:

- `question_id`
- `question`
- `answer`

For `humaneval`, these fields map as:

- `question_id = task_id`
- `question = prompt`
- `answer = canonical_solution`

The dataset item must also preserve eval-only metadata:

- `task_id`
- `prompt`
- `canonical_solution`
- `test`
- `entry_point`

This lets training remain simple while giving evaluation the official problem definition it needs.

### Training Target

Training uses `completion only`.

That means:

- model input is the HumanEval `prompt`
- supervised target is `canonical_solution`
- the target does not repeat the prompt itself

This is the closest match to the official completion format and keeps the train objective aligned with the generation artifact used by evaluation.

## Evaluation Design

### Primary Metric

For `humaneval`, the primary metric is official `pass@k`.

The current exact-match path is not valid for this task and must be bypassed entirely when `training.task == "humaneval"`.

### Generation Flow

Evaluation must generate multiple completions per HumanEval problem. The configuration needs explicit controls for:

- number of samples per task
- sampling temperature
- `pass@k` values to report
- `do_sample`

The generated artifacts must be written in the official JSONL sample format:

```json
{"task_id": "HumanEval/0", "completion": "    return x + 1"}
```

### Harness Integration

After generation, evaluation must call the official HumanEval functional-correctness harness and ingest its results.

The implementation should:

- write `samples.jsonl`
- run official correctness evaluation on that file
- read back per-sample results
- aggregate `pass@k`
- save a repository-native summary JSON alongside the raw official artifacts

### Security And Execution Note

Official HumanEval evaluation runs untrusted generated code. The repository must treat this as an explicit execution-risk boundary.

The code path should fail loudly if:

- the `human_eval` package is unavailable
- execution support is not enabled in the installed harness

It must not silently fall back to string matching.

## Output Artifacts

For `humaneval`, the output directory should include at least:

- repository-native eval summary JSON
- raw generated `samples.jsonl`
- official results JSONL produced by the harness
- optional agent logs for debugging

The repository-native eval summary should report:

- task name
- `num_samples_per_task`
- requested `k` values
- reported `pass@k`
- paths to generated sample/result files
- optional per-task sample summaries

This keeps the official artifact intact while preserving the current repository conventions.

## Configuration Design

Add a dedicated experiment config such as `configs/experiments/humaneval_5agent.yaml`.

It should mirror the existing `gsm8k` layout but set HumanEval-specific knobs, including:

- `training.task: humaneval`
- `evaluation.metric: pass_at_k`
- `evaluation.num_samples_per_task`
- `evaluation.pass_at_k`
- `evaluation.temperature`
- `evaluation.do_sample: true`

The rest of the training structure can stay close to the existing 5-agent setup unless runtime constraints require separate tuning.

## Code Changes

### `data/dataset.py`

- add `humaneval` to `TASK_CONFIGS`
- support its field mapping
- preserve eval-only metadata on each item

### `src/cli/train.py`

- ensure the collate path can ignore extra metadata for normal training
- no special-case logic should be required beyond loading the new task

### `src/cli/evaluate.py`

- branch on `humaneval`
- skip `extract_answer(... ) == gold`
- generate multiple completions per task
- write official sample JSONL
- call official harness
- ingest `pass@k` outputs and persist summary files

### `src/utils/answer_extraction.py`

- no extraction-based scoring for `humaneval`
- optional defensive behavior can be added, but this module should not be part of the HumanEval metric path

## Test Plan

Minimum required tests:

1. dataset test
   - verify `humaneval` field mapping
   - verify `task_id`, `prompt`, `canonical_solution`, `test`, and `entry_point` survive loading

2. evaluation plumbing test
   - verify `humaneval` uses the code-eval branch instead of exact-match scoring
   - verify `samples.jsonl` shape matches official expectations

3. results parsing test
   - verify official harness output is converted into repository-native `pass@k` metrics correctly

Tests should avoid executing arbitrary code in unit-test context. Harness invocation should therefore be wrapped behind mockable functions.

## Documentation Impact

The following main docs must be updated when implementation lands because runtime behavior will materially change:

- `docs/training_pipeline.md`
- `docs/method.md`
- `docs/agent_workflow.md`
- `docs/prompt_flow.md`
- `docs/ours_json_log_format.md`

Repository policy also references `docs/README.md`, but that file is currently absent in this workspace. The implementation session should either add it or explicitly resolve that documentation gap.

## Risks

### Metric Mismatch Risk

If evaluation accidentally keeps the old exact-match path for `humaneval`, the reported metric will be invalid.

### Sampling Risk

`pass@k` requires multiple completions per task. Reusing the current one-shot eval loop without adaptation will produce incomplete or misleading results.

### Environment Risk

The official harness may not be installed or may require explicit execution enablement. This must be surfaced as a hard error with clear instructions.

### Performance Risk

HumanEval eval can be slower than `gsm8k` because each task may require many generated samples and code execution. The config should make sample count explicit rather than hidden in code defaults.

## Recommendation

Implement HumanEval as the first task-specific official-harness eval path inside the current CLI structure, without attempting a broader task-plugin redesign.

This gives the repository a correct code benchmark pipeline quickly while keeping the current user workflow familiar and minimizing disruption to the existing GSM8K setup.
