# AM DeepSeek R1 Source Filter Design

**Date:** 2026-03-26

## Goal

Add source-based sample selection for `am_deepseek_r1_distilled` without changing the existing `data/am_deepseek_r1_distilled/train.jsonl` file.

## User Constraints

- `train.jsonl` must keep its current row format and remain the training source of truth.
- The `source` label cannot be trusted from `train.jsonl`; it must be rebuilt from the original Hugging Face dataset.
- The user wants YAML to choose which `source` values are allowed for training.
- The result should not require running `train.sh` as part of this task.

## Approved Design

### Data Artifacts

Keep the existing prepared training file:

```text
data/am_deepseek_r1_distilled/train.jsonl
```

Add one new local index artifact:

```text
data/am_deepseek_r1_distilled/source_index.jsonl
```

Each line in `source_index.jsonl` will contain:

```json
{"question_id": "am-r1-<hash>", "source": "<hf source string>"}
```

The `question_id` generation must use the exact same deterministic rule already used for `train.jsonl`, so the two files can be joined locally without rewriting training rows.

### Source Extraction

The source metadata must be rebuilt from the original Hugging Face rows, not from `train.jsonl`.

When a raw sample is downloaded again during preparation:

- compute the stable `question_id`
- extract the sample `source`
- write one index row to `source_index.jsonl`

The implementation should read the same raw subsets already used for `train.jsonl` generation:

- `am_0.5M`
- `am_0.9M`

### Runtime Loading

The dataset loader for `am_deepseek_r1_distilled` will keep reading normalized training content from:

```text
data/am_deepseek_r1_distilled/train.jsonl
```

If no YAML `sources` filter is provided, loader behavior remains unchanged: read all rows from `train.jsonl`.

If YAML provides a `sources` list, the loader must:

1. read `source_index.jsonl`
2. collect all `question_id` values whose `source` is in the requested set
3. filter the existing `train.jsonl` dataset to those `question_id` values only

This keeps the training text path stable while allowing source-based filtering.

### YAML Contract

The approved configuration key lives under `training`:

```yaml
training:
  task: "am_deepseek_r1_distilled"
  sources:
    - "am-0309"
    - "natural_reasoning"
```

Semantics:

- missing `sources` or empty list: train on all rows
- non-empty `sources`: train only on rows whose `question_id` is present in `source_index.jsonl` with one of those sources

This new key should be treated as task-specific optional config; it should not affect other datasets.

## Error Handling

If `training.sources` is set but `source_index.jsonl` is missing, fail with a clear error telling the user to rerun the prepare script.

If requested sources do not exist in the index, fail with a clear error that lists the missing sources.

If the index contains duplicate `question_id -> source` conflicts, fail rather than silently choosing one.

## Testing Strategy

Add tests for:

- prepare step writes `source_index.jsonl` from raw HF-like rows
- loader defaults to full `train.jsonl` when no `sources` are configured
- loader filters to requested `sources` when `source_index.jsonl` is present
- loader raises clear errors when the index is missing or requested sources are unknown
- config loading preserves `training.sources`

## Files Expected To Change

- Modify `scripts/prepare_am_deepseek_r1_distilled.py`
- Modify `src/data/am_deepseek_r1_distilled.py`
- Modify `src/data/base.py`
- Modify `src/data/factory.py`
- Modify `src/cli/train.py`
- Modify `tests/test_prepare_am_deepseek_r1_distilled.py`
- Modify `tests/test_dataset_ids.py`
- Modify `tests/test_config.py`
- Modify `docs/data/am_deepseek_r1_distilled.md`
- Modify `docs/training_pipeline.md`

## Success Criteria

This design is complete when:

- `train.jsonl` is left untouched
- the prepare workflow can regenerate `source_index.jsonl` from Hugging Face raw rows
- YAML can specify `training.sources`
- the `am_deepseek_r1_distilled` loader filters by source using the index file
- tests prove both default and filtered behavior
