"""
Answer extraction utilities for different task types.

Each task type has a different way of extracting the final answer
from the model's generated text.

Extensibility:
  - Add new extraction functions for new task types.
  - Register them in EXTRACTORS dict.
"""

import re


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
        return match.group(1).replace(",", "")

    # Try "answer is <number>" pattern
    match = re.search(r"answer\s+is\s+(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

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


def _extract_math(text: str) -> str:
    """Extract answer from MATH-style output with \\boxed{}."""
    import re
    match = re.search(r'\\boxed\{([^}]*)\}', text)
    if match:
        return match.group(1).strip()
    # Try "answer is" pattern
    match = re.search(r'answer\s+is\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip().split('\n')[-1].strip()


def _extract_default(text: str) -> str:
    """Default: return stripped text."""
    return text.strip()


# Registry of extractors
EXTRACTORS = {
    "gsm8k": _extract_gsm8k,
    "arc": _extract_arc,
    "arc_easy": _extract_arc,
    "arc_challenge": _extract_arc,
    "medqa": _extract_arc,  # also multiple choice
    "math": _extract_math,
}
