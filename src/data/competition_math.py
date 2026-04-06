"""Competition Math dataset helpers and config."""

from .math import _extract_boxed_answer


def _load_hf_dataset(dataset_name: str, subset: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, subset, split=split)


def build_task_configs() -> dict:
    return {
        "competition_math": {
            "loader": lambda split: _load_hf_dataset("qwedsacf/competition_math", None, "train"),
            "allowed_splits": ("train",),
            "question_id_field": "problem",
            "question_field": "problem",
            "answer_field": "solution",
            "answer_extractor": _extract_boxed_answer,
            "extra_fields": ("level", "type"),
        }
    }
