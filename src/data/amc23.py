"""AMC23 dataset helpers and config."""

import json
from pathlib import Path

LOCAL_DATA_DIR = Path("/mnt/3fs/data/yfzhang/cache/local_datasets")


def _load_local(split: str):
    path = LOCAL_DATA_DIR / f"amc23_{split}.json"
    with open(path) as f:
        return json.load(f)


def build_task_configs() -> dict:
    return {
        "amc23": {
            "loader": _load_local,
            "allowed_splits": ("test",),
            "question_id_field": "id",
            "question_field": "question",
            "answer_field": "answer",
        }
    }
