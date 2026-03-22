"""HumanEval dataset helpers and config."""


def _load_hf_dataset(dataset_name: str, subset: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, subset, split=split)


def _apply_humaneval_split(raw, split: str):
    total = len(raw)
    train_cutoff = int(total * 0.6)

    split_to_indices = {
        "train": list(range(0, train_cutoff)),
        "test": list(range(train_cutoff, total)),
    }
    if split not in split_to_indices:
        raise ValueError(
            "Unsupported split for humaneval. "
            "Use 'train' or 'test' for the local 60/40 debug split."
        )

    return raw.select(split_to_indices[split])


def build_task_configs() -> dict:
    return {
        "humaneval": {
            "loader": lambda split: _load_hf_dataset("openai_humaneval", None, "test"),
            "split_handler": _apply_humaneval_split,
            "allowed_splits": ("train", "test"),
            "question_id_field": "task_id",
            "question_field": "prompt",
            "answer_field": "canonical_solution",
            "extra_fields": ("task_id", "prompt", "canonical_solution", "test", "entry_point"),
        }
    }
