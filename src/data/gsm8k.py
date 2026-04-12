"""GSM8K dataset helpers and config."""
import json
import re
from pathlib import Path

LOCAL_DATA_DIR = Path("/mnt/3fs/data/yfzhang/cache/local_datasets")


def _load_local(split: str):
    path = LOCAL_DATA_DIR / f"gsm8k_{split}.json"
    with open(path) as f:
        return json.load(f)


def _format_gsm8k_answer(answer_text: str) -> str:
    cleaned = re.sub(r"<<[^>]*>>", "", answer_text)
    return cleaned.strip()


def _format_gsm8k_question(item: dict) -> str:
    return item["question"] + "\n\nPlease put your final answer in \\boxed{}."


def build_task_configs() -> dict:
    return {
        "gsm8k": {
            "loader": _load_local,
            "question_id_field": "id",
            "question_field": "question",
            "answer_field": "answer",
            "answer_extractor": _format_gsm8k_answer,
            "train_label_extractor": _format_gsm8k_answer,
            "question_formatter": _format_gsm8k_question,
        }
    }
