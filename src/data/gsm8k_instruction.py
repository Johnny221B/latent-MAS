"""GSM8K-Instruction dataset helpers and config (notefill/gsm8k-instruction)."""

import json
from pathlib import Path

LOCAL_DATA_DIR = Path("/mnt/3fs/data/yfzhang/cache/local_datasets")


def _load_local(split: str):
    path = LOCAL_DATA_DIR / "gsm8k_instruction_train.json"
    with open(path) as f:
        data = json.load(f)
    if split == "train":
        return [item for item in data if item.get("split", "train") == "train"]
    raise ValueError(f"Unsupported split: {split}")


def _format_question(item: dict) -> str:
    instruction = item.get("instruction", "").strip()
    question = item["input"].strip()
    parts = []
    if instruction:
        parts.append(instruction)
    parts.append(question)
    parts.append("Please put your final answer in \\boxed{}.")
    return "\n\n".join(parts)


def _format_raw_answer(item: dict) -> str:
    output = item["output"].strip()
    final_answer = str(item["final_answer"]).strip()
    return f"{output}\n\nThe answer is $\\boxed{{{final_answer}}}$."


def build_task_configs() -> dict:
    return {
        "gsm8k_instruction": {
            "loader": _load_local,
            "allowed_splits": ("train",),
            "question_id_field": "id",
            "question_field": "input",
            "answer_field": "final_answer",
            "raw_answer_field": "output",
            "question_formatter": _format_question,
            "raw_answer_formatter": _format_raw_answer,
        }
    }
