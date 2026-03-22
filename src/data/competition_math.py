"""Competition Math dataset helpers and config."""

import re


def _load_hf_dataset(dataset_name: str, subset: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, subset, split=split)


def _strip_latex_wrappers(value: str) -> str:
    cleaned = str(value).strip()
    wrappers = (
        r"\(",
        r"\)",
        "$",
    )
    for wrapper in wrappers:
        cleaned = cleaned.replace(wrapper, "")
    return cleaned.strip()


def _extract_braced_content(text: str, start_index: int) -> str | None:
    brace_start = text.find("{", start_index)
    if brace_start < 0:
        return None

    depth = 0
    content = []
    for char in text[brace_start + 1:]:
        if char == "{":
            depth += 1
            content.append(char)
            continue
        if char == "}":
            if depth == 0:
                return "".join(content).strip()
            depth -= 1
            content.append(char)
            continue
        content.append(char)
    return None


def _extract_competition_math_answer(answer_text: str) -> str:
    text = str(answer_text).strip()

    boxed_match = re.search(r"\\boxed", text)
    if boxed_match:
        boxed = _extract_braced_content(text, boxed_match.start())
        if boxed:
            return _strip_latex_wrappers(boxed)

    final_answer_patterns = (
        r"(?:final answer|answer)\s*(?:is|:)\s*([^\n]+)",
        r"=\s*([^\n]+)\s*$",
    )
    for pattern in final_answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _strip_latex_wrappers(match.group(1))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return _strip_latex_wrappers(lines[-1])
    return ""


def build_task_configs() -> dict:
    return {
        "competition_math": {
            "loader": lambda split: _load_hf_dataset("qwedsacf/competition_math", None, "train"),
            "allowed_splits": ("train",),
            "question_id_field": "problem",
            "question_field": "problem",
            "answer_field": "solution",
            "answer_extractor": _extract_competition_math_answer,
            "extra_fields": ("level", "type"),
        }
    }
