"""AIME 2025 dataset helpers and config."""

import json
from pathlib import Path

LOCAL_DATA_DIR = Path("/mnt/3fs/data/yfzhang/cache/local_datasets")


def _load_local(split: str):
    # AIME 2025 only has a "train" split on HF, we use it as "test"
    path = LOCAL_DATA_DIR / f"aime2025_train.json"
    with open(path) as f:
        return json.load(f)


def build_task_configs() -> dict:
    return {
        "aime2025": {
            "loader": _load_local,
            "allowed_splits": ("test", "train"),
            "question_id_field": "problem_idx",
            "question_field": "problem",
            "answer_field": "answer",
        }
    }
