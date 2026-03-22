"""
Answer extraction utilities for different task types.

Each task type has a different way of extracting the final answer
from the model's generated text.

Extensibility:
  - Add new extraction functions for new task types.
  - Register them in EXTRACTORS dict.
"""

import re


def _normalize_numeric_text(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    if re.fullmatch(r"-?\d+\.", cleaned):
        cleaned = cleaned[:-1]
    if re.fullmatch(r"-?\d+\.0+", cleaned):
        cleaned = cleaned.split(".", 1)[0]
    return cleaned


def extract_answer(text: str, task_type: str = "gsm8k") -> str:
    """Extract the final answer from generated text.

    Args:
        text: raw generated text from the terminal agent
        task_type: one of ["gsm8k", "arc", "medqa", "code"]

    Returns:
        extracted answer string
    """
    extractor = EXTRACTORS.get(task_type, _extract_default)
    return extractor(text)


def _extract_gsm8k(text: str) -> str:
    """Extract numeric answer from GSM8K-style output.

    Looks for patterns like "#### 42" or "The answer is 42".
    """
    # Try "#### <number>" pattern first
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return _normalize_numeric_text(match.group(1))

    # Try "answer is <number>" pattern
    match = re.search(r"answer\s+is\s+(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return _normalize_numeric_text(match.group(1))

    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return _normalize_numeric_text(numbers[-1])

    return text.strip()


def _extract_arc(text: str) -> str:
    """Extract multiple-choice answer (A/B/C/D) from ARC-style output."""
    # Look for explicit answer markers
    match = re.search(r"(?:answer|choice)\s*(?:is|:)\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for standalone letter at the end
    match = re.search(r"\b([A-D])\s*$", text.strip())
    if match:
        return match.group(1).upper()

    return text.strip()[:1].upper()


def _extract_default(text: str) -> str:
    """Default: return stripped text."""
    return text.strip()


def _extract_competition_math(text: str) -> str:
    boxed_match = re.search(r"\\boxed\s*{([^{}]+)}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    match = re.search(r"(?:final answer|answer)\s*(?:is|:)\s*([^\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip("$")

    return _extract_default(text)


# Registry of extractors
EXTRACTORS = {
    "gsm8k": _extract_gsm8k,
    "arc": _extract_arc,
    "arc_easy": _extract_arc,
    "arc_challenge": _extract_arc,
    "medqa": _extract_arc,  # also multiple choice
    "competition_math": _extract_competition_math,
}
