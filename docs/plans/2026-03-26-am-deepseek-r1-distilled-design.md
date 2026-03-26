# AM DeepSeek R1 Distilled Dataset Design

**Date:** 2026-03-26

## Goal

Add a new training dataset task for `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` so the repository can supervise the terminal agent on the full assistant output, including both the `<think>...</think>` and `<answer>...</answer>` sections.

## Current Context

The repository currently normalizes all datasets through `src/data/base.py` and `src/data/factory.py`. Each task maps raw examples into a common shape:

```text
{
  "question_id": ...,
  "question": ...,
  "answer": ...,
}
```

Training then applies teacher forcing on the `answer` text. For `competition_math`, that means supervising on the full `solution` field. For this new dataset, the supervision target must instead be the full assistant response text, not just a final extracted answer.

## Source Dataset

- Hugging Face dataset: `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`
- Relevant subsets: `am_0.5M`, `am_0.9M`
- The dataset card currently describes examples with:
  - `messages`
  - `info.think_content`
  - `info.answer_content`
  - additional metadata such as `source`, `reference_answer`, and `test_case`

The assistant `messages` entry already contains the complete response with `<think>` and `<answer>` tags. That field should remain the training target verbatim.

## Approved Design

### Task Identity

- Add a new task named `am_deepseek_r1_distilled`
- Keep it separate from `competition_math`

This avoids overloading an existing task whose current semantics are `problem -> solution`. The new task has different supervision semantics because the label is the full assistant response format.

### Split Semantics

- Support only `train`
- Do not invent a synthetic `test` split

This dataset is being used as training supervision data rather than a benchmark with an official evaluation partition. Any train-time monitoring should continue to use the repository's existing `training_probe` flow rather than a fake dataset split.

### Dataset Composition

- Load both `am_0.5M` and `am_0.9M`
- Concatenate them into one task-level dataset

The task should behave like one logical training source while still retaining per-example subset metadata for traceability.

### Field Mapping

- `question <- user message content`
- `answer <- assistant message content`

The assistant content must remain the complete text including:

```text
<think>...</think><answer>...</answer>
```

No answer extraction or post-processing should run on this task.

### Metadata Preservation

Preserve the following fields in the normalized sample when available:

- `subset`
- `messages`
- `source`
- `reference_answer`
- `test_case`
- `think_content`
- `answer_content`

This keeps future analysis and ablations possible without changing the core dataset contract later.

### Question ID Strategy

Do not use the raw question text as `question_id`. Instead generate a stable digest from:

- subset name
- user content
- assistant content

Recommended format:

```text
am-r1-<sha1-prefix>
```

This keeps IDs short, deterministic, and stable across reloads while avoiding collisions between the two subsets.

## Files To Change

- Create `src/data/am_deepseek_r1_distilled.py`
- Modify `src/data/factory.py`
- Modify `tests/test_dataset_ids.py`
- Create `docs/data/am_deepseek_r1_distilled.md`
- Modify `docs/data/README.md`
- Modify `docs/training_pipeline.md`
- Modify `docs/README.md`

## Testing Strategy

Follow TDD:

1. Add failing tests that mock the Hugging Face loaders for both subsets
2. Verify the task appears in the registry
3. Verify the merged dataset returns both subset rows
4. Verify `question` and `answer` extraction from `messages`
5. Verify `answer` retains the full `<think>` and `<answer>` tagged assistant response
6. Verify metadata passthrough
7. Verify generated `question_id` is deterministic and short

## Documentation Impact

The new task changes supported runtime dataset behavior, so the canonical runtime docs need to mention it. The core training mechanism does not change, so the update scope should stay focused on dataset and training-pipeline descriptions rather than the broader method docs unless implementation reveals a prompt-flow change.
