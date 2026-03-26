# AM DeepSeek R1 Local Preprocess Design

**Date:** 2026-03-26

## Goal

Change the `am_deepseek_r1_distilled` data path so `train.sh` no longer downloads the Hugging Face dataset at training time. Instead, the data should be downloaded once, normalized into a local repository file, and then loaded from that local file by the training dataloader.

## Problem Statement

The current implementation loads `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` directly inside the dataset loader. On `c1103a-s5`, that path fails before training begins with:

```text
ValueError: Compression type zstd not supported
```

Investigation on the remote node showed:

- GPUs are available and model loading succeeds
- failure happens only when dataset generation starts
- `.venv` is missing `zstandard`

Even if that missing dependency is fixed, training-time online data loading is still the wrong shape for this workflow because:

- it mixes data preparation and training startup
- it makes failures harder to localize
- it adds unnecessary remote-network and format-coupling risk

## Approved Design

### Data Preparation Model

Add a dedicated preparation step that downloads both Hugging Face subsets:

- `am_0.5M`
- `am_0.9M`

and writes one normalized local file:

```text
data/am_deepseek_r1_distilled/train.jsonl
```

The normalized rows should already match the training dataset contract.

### Normalized Row Shape

Each output line in `train.jsonl` should include:

- `question_id`
- `question`
- `answer`
- `subset`
- `messages`
- `source`
- `reference_answer`
- `test_case`
- `think_content`
- `answer_content`

Semantics:

- `question` comes from the `user` message content
- `answer` is the full `assistant` message content
- `<think>` and `<answer>` tags must remain intact
- `question_id` remains deterministic and hash-based

### Training Loader Behavior

The runtime dataset loader should stop reading Hugging Face directly.

Instead it should:

1. open `data/am_deepseek_r1_distilled/train.jsonl`
2. load it as a local dataset
3. return normalized samples to the existing `MultiAgentDataset`

If the file is missing, it must fail with a clear message telling the user to run the prepare script first.

### Script Boundary

Create a dedicated preparation script:

```text
scripts/prepare_am_deepseek_r1_distilled.py
```

Responsibilities:

- download raw subsets from Hugging Face
- normalize rows
- write local `jsonl`
- create parent directory if needed
- optionally skip rewriting when the output already exists unless forced

Non-responsibilities:

- starting training
- mutating experiment config
- auto-running from `train.sh`

### Train Flow

The intended runtime flow becomes:

1. run the prepare script once on the target machine
2. run `bash scripts/train.sh`

This keeps data preparation and training startup clearly separated.

## Files To Change

- Create `scripts/prepare_am_deepseek_r1_distilled.py`
- Modify `src/data/am_deepseek_r1_distilled.py`
- Modify `tests/test_dataset_ids.py`
- Modify `tests/test_config.py` only if config semantics need additional coverage
- Modify `docs/data/am_deepseek_r1_distilled.md`
- Modify `docs/training_pipeline.md`

## Testing Strategy

Follow TDD:

1. add loader tests for local `jsonl` reading
2. add a failure test for missing local file with a helpful error
3. add unit coverage for the prepare script normalization logic
4. run focused test slices locally
5. run the prepare script on `c1103a-s5`
6. rerun `bash scripts/train.sh` on `c1103a-s5`

## Success Criteria

The task is complete only when:

- local tests for the new data path pass
- the prepare script successfully writes `data/am_deepseek_r1_distilled/train.jsonl`
- `bash scripts/train.sh` starts training on `c1103a-s5` and gets past dataset loading
