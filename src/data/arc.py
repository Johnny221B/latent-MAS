"""ARC dataset helpers and config."""


def _load_hf_dataset(dataset_name: str, subset: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, subset, split=split)


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


def _extract_arc_answer(answer_key: str) -> str:
    return str(answer_key).strip()


def build_task_configs() -> dict:
    return {
        "arc_easy": {
            "loader": lambda split: _load_hf_dataset("allenai/ai2_arc", "ARC-Easy", split),
            "question_id_field": "id",
            "question_field": "question",
            "question_formatter": _format_arc_question,
            "answer_field": "answerKey",
            "answer_extractor": _extract_arc_answer,
        },
        "arc_challenge": {
            "loader": lambda split: _load_hf_dataset("allenai/ai2_arc", "ARC-Challenge", split),
            "question_id_field": "id",
            "question_field": "question",
            "question_formatter": _format_arc_question,
            "answer_field": "answerKey",
            "answer_extractor": _extract_arc_answer,
        },
    }
