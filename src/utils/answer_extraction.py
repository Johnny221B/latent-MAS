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

    Priority: <answer> tag > #### pattern > "answer is" pattern > last number.
    """
    # Try <answer> tag first (matches training data format)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
        if numbers:
            return _normalize_numeric_text(numbers[-1])
        return answer_text

    # Try "#### <number>" pattern
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


_ARC_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def _arc_parse_letter_or_num(s: str) -> str | None:
    """Try to extract a choice letter/number from a short string like 'D', 'D.', '3', 'D. ovary cells'."""
    s = s.strip()
    m = re.match(r"^([A-Ea-e])\b", s)
    if m:
        return m.group(1).upper()
    m = re.match(r"^([1-5])\b", s)
    if m:
        return _ARC_NUM_TO_LETTER.get(m.group(1), m.group(1))
    return None


def _extract_arc(text: str) -> str:
    """Extract multiple-choice answer (A/B/C/D/E) from ARC-style output."""
    # 1. <answer> tag (highest priority, matches training format)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        parsed = _arc_parse_letter_or_num(match.group(1))
        if parsed:
            return parsed

    # 2. "answer is D" / "choice is D" style markers
    match = re.search(r"(?:answer|choice)\s*(?:is|:)\s*([A-Ea-e])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?:answer|choice)\s*(?:is|:)\s*([1-5])\b", text, re.IGNORECASE)
    if match:
        return _ARC_NUM_TO_LETTER.get(match.group(1), match.group(1))

    # 3. Standalone letter at end of text
    match = re.search(r"\b([A-Ea-e])\s*\.?\s*$", text.strip())
    if match:
        return match.group(1).upper()

    # 4. Standalone number (1-5) at end of text
    match = re.search(r"\b([1-5])\s*\.?\s*$", text.strip())
    if match:
        return _ARC_NUM_TO_LETTER.get(match.group(1), match.group(1))

    # 5. Last <answer> tag content in text (without closing tag)
    match = re.search(r"<answer>\s*([A-Ea-e1-5])", text, re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        return _ARC_NUM_TO_LETTER.get(val, val.upper())

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


def _extract_boxed_nested(text: str) -> str | None:
    r"""Extract content from the last \boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = text.index("{", idx)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i].strip()
    return None


def _is_latex_expression(text: str) -> bool:
    """Check if text contains LaTeX math commands (not just plain numbers)."""
    return bool(re.search(
        r"\\(?:d?t?frac|sqrt|infty|pi|cdot|times|left|right|text|mathrm|mathit|"
        r"mathbf|overline|underline|hat|bar|vec|tilde|log|ln|sin|cos|tan|lim|sum|"
        r"prod|int|binom|pmod|equiv|approx|neq|leq|geq|pm|mp|cap|cup|subset|"
        r"supset|in|notin|emptyset|forall|exists|alpha|beta|gamma|delta|epsilon|"
        r"theta|lambda|mu|sigma|omega|begin|end|quad)",
        text,
    ))


def _extract_math(text: str) -> str:
    """Extract answer from MATH-style output.

    Priority: <answer> tag > \\boxed{} (nested) > "answer is" pattern > last number.
    If the text is already a LaTeX expression (e.g. pre-extracted from \\boxed{}),
    return it as-is instead of falling back to last-number extraction.
    """
    # 1. <answer> tag
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        # If the <answer> tag contains \boxed{}, extract from it
        boxed = _extract_boxed_nested(inner)
        if boxed:
            return boxed
        return inner

    # 2. \boxed{} with nested braces
    boxed = _extract_boxed_nested(text)
    if boxed:
        return boxed

    # 3. "#### <answer>" pattern (GSM8K-style, some models output this)
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()

    # 4. "answer is ..." / "final answer is ..."
    match = re.search(r"(?:final answer|answer)\s*(?:is|:)\s*([^\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip("$").strip(".")

    # 5. If text looks like a LaTeX expression already (e.g. gold answer
    #    pre-extracted from \boxed{}), return as-is — do NOT reduce to last number.
    if _is_latex_expression(text):
        return text.strip().strip("$")

    # 6. Fallback: last number
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return _normalize_numeric_text(numbers[-1])

    return text.strip()


def _extract_humaneval(text: str) -> str:
    """Extract code completion from HumanEval-style output.

    Priority: last <answer> tag > last ```python``` block > raw text.
    """
    # 1. Last <answer> tag (may contain full function)
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    # 2. Last ```python block
    matches = list(re.finditer(r"```python\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    # 3. Last ``` block (any language)
    matches = list(re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    return text.strip()


def _normalize_math_str(s: str) -> str:
    """Normalize a math answer string for comparison."""
    s = s.strip().strip("$").strip(".")
    # \dfrac -> \frac, \tfrac -> \frac
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    # Remove \text{}, \mathrm{}, \mathit{} wrappers
    s = re.sub(r"\\(?:text|mathrm|mathit|textbf)\s*\{([^}]*)\}", r"\1", s)
    # Remove \left \right
    s = re.sub(r"\\(?:left|right)\s*", "", s)
    # Remove spaces
    s = s.replace(" ", "")
    return s


def _try_parse_sympy(expr_str: str):
    """Try to parse a math expression string with sympy. Returns None on failure."""
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(expr_str)
    except Exception:
        pass
    try:
        from sympy import sympify
        cleaned = expr_str.replace("\\", "")
        return sympify(cleaned, rational=True)
    except Exception:
        return None


def _is_numeric_equal(a: str, b: str) -> bool:
    """Check if two strings represent the same number (e.g. '.5' vs '0.5' vs '1/2')."""
    try:
        from fractions import Fraction
        fa = Fraction(a.replace(",", ""))
        fb = Fraction(b.replace(",", ""))
        return fa == fb
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-9
    except (ValueError, OverflowError):
        return False


def math_is_equivalent(pred: str, gold: str) -> bool:
    """Check if two math answers are equivalent.

    Tries in order:
    1. Normalized string equality
    2. Numeric equality (handles .5 vs 1/2 vs 0.5)
    3. Sympy symbolic equality (handles \\frac{x+2}{7} vs (x+2)/7)
    """
    # 1. Normalized string match
    np, ng = _normalize_math_str(pred), _normalize_math_str(gold)
    if np == ng:
        return True

    # 2. Numeric equality
    if _is_numeric_equal(np, ng):
        return True

    # 3. Sympy symbolic equality
    sp = _try_parse_sympy(pred)
    sg = _try_parse_sympy(gold)
    if sp is not None and sg is not None:
        try:
            from sympy import simplify, Eq
            diff = simplify(sp - sg)
            if diff == 0 or diff.is_zero:
                return True
            if simplify(Eq(sp, sg)) is True:
                return True
        except Exception:
            pass

    return False


# Tasks that should use math_is_equivalent for comparison
MATH_EQUIVALENT_TASKS = {
    "math", "math500", "minerva_math", "competition_math",
    "aime2025", "amc23",
}


# Registry of extractors
EXTRACTORS = {
    "gsm8k": _extract_gsm8k,
    "arc": _extract_arc,
    "arc_easy": _extract_arc,
    "arc_challenge": _extract_arc,
    "medqa": _extract_arc,  # also multiple choice
    "aime2025": _extract_math,
    "amc23": _extract_math,
    "competition_math": _extract_competition_math,
    "math": _extract_math,
    "math500": _extract_math,
    "minerva_math": _extract_math,
    "humaneval": _extract_humaneval,
}
