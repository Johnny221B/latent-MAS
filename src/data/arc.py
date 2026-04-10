"""ARC dataset helpers and config."""

import json
from pathlib import Path

LOCAL_DATA_DIR = Path("/mnt/3fs/data/yfzhang/cache/local_datasets")


def _load_arc(subset_tag: str, split: str):
    """Load ARC from local JSON, fall back to HuggingFace."""
    local_path = LOCAL_DATA_DIR / f"{subset_tag}_{split}.json"
    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)
    from datasets import load_dataset
    hf_subset = "ARC-Easy" if "easy" in subset_tag else "ARC-Challenge"
    return load_dataset("allenai/ai2_arc", hf_subset, split=split)


def _iter_arc_choices(raw_choices) -> list[tuple[str, str]]:
    if isinstance(raw_choices, dict):
        labels = raw_choices.get("label", ())
        texts = raw_choices.get("text", ())
        return [
            (str(label).strip().upper(), str(text).strip())
            for label, text in zip(labels, texts)
        ]

    if isinstance(raw_choices, list):
        pairs = []
        for choice in raw_choices:
            if not isinstance(choice, dict):
                continue
            label = str(choice.get("label", "")).strip().upper()
            text = str(choice.get("text", "")).strip()
            if label and text:
                pairs.append((label, text))
        return pairs

    return []


def _format_arc_question(item: dict) -> str:
    question = str(item.get("question", "")).strip()
    choice_lines = [
        f"{label}. {text}"
        for label, text in _iter_arc_choices(item.get("choices"))
        if label and text
    ]
    if not choice_lines:
        return question
    return f"{question}\n\nChoices:\n" + "\n".join(choice_lines)


_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def _extract_arc_answer(answer_key: str) -> str:
    key = str(answer_key).strip()
    return _NUM_TO_LETTER.get(key, key).upper()


def build_task_configs() -> dict:
    return {
        "arc_easy": {
            "loader": lambda split: _load_arc("arc_easy", split),
            "question_id_field": "id",
            "question_field": "question",
            "question_formatter": _format_arc_question,
            "answer_field": "answerKey",
            "answer_extractor": _extract_arc_answer,
        },
        "arc_challenge": {
            "loader": lambda split: _load_arc("arc_challenge", split),
            "question_id_field": "id",
            "question_field": "question",
            "question_formatter": _format_arc_question,
            "answer_field": "answerKey",
            "answer_extractor": _extract_arc_answer,
        },
    }
