"""Math dataset helpers — provides _extract_boxed_answer for other data modules."""

from ..utils.answer_extraction import extract_answer as _extract_answer


def _extract_boxed_answer(text: str) -> str | None:
    """Extract the final answer from a math solution (boxed or otherwise)."""
    return _extract_answer(text, task_type="competition_math")


def build_task_configs() -> dict:
    return {}
