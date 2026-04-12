"""Competition Math dataset helpers and config."""

import json
from pathlib import Path

from .math import _extract_boxed_answer

LOCAL_DATA_DIR = Path("/mnt/3fs/data/yfzhang/cache/local_datasets")


def _load_local(split: str):
    path = LOCAL_DATA_DIR / f"competition_math_train.json"
    with open(path) as f:
        return json.load(f)


def _format_competition_math_question(item: dict) -> str:
    return item["problem"] + "\n\nPlease put your final answer in \\boxed{}."


def build_task_configs() -> dict:
    return {
        "competition_math": {
            "loader": _load_local,
            "allowed_splits": ("train",),
            "question_id_field": "problem",
            "question_field": "problem",
            "answer_field": "solution",
            "answer_extractor": _extract_boxed_answer,
            "question_formatter": _format_competition_math_question,
            "extra_fields": ("level", "type"),
        }
    }
