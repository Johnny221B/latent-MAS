"""MetaMathQA dataset helpers and config."""

import json
from pathlib import Path

LOCAL_DATA_DIR = Path("/mnt/3fs/data/yfzhang/cache/local_datasets")


def _load_local(split: str):
    if split != "train":
        raise ValueError("MetaMathQA only supports the 'train' split")
    path = LOCAL_DATA_DIR / "metamathqa_train.json"
    with open(path) as f:
        return json.load(f)


def _format_question(item: dict) -> str:
    return item["query"] + "\n\nPlease put your final answer in \\boxed{}."


def build_task_configs() -> dict:
    return {
        "metamathqa": {
            "loader": _load_local,
            "allowed_splits": ("train",),
            "question_id_field": "query",
            "question_field": "query",
            "answer_field": "response",
            "question_formatter": _format_question,
        }
    }
