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

    Priority: \\boxed{} > <answer> tag > #### pattern > "answer is" pattern > last number.
    """
    # Try \boxed{} first (highest priority when boxed prompt is used)
    match = re.search(r"\\boxed\s*\{([^}]*)\}", text)
    if match:
        val = match.group(1).strip()
        numbers = re.findall(r"-?[\d,]+\.?\d*", val)
        if numbers:
            return _normalize_numeric_text(numbers[-1])
        return val

    # Try <answer> tag
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
    Strips the function signature if the model re-generated it.
    """
    code = None

    # 1. Last <answer> tag (may contain full function)
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL))
    if matches:
        code = matches[-1].group(1).strip()

    # 2. Last ```python block
    if code is None:
        matches = list(re.finditer(r"```python\s*\n(.*?)```", text, re.DOTALL))
        if matches:
            code = matches[-1].group(1).strip()

    # 3. Last ``` block (any language)
    if code is None:
        matches = list(re.finditer(r"```\s*\n(.*?)```", text, re.DOTALL))
        if matches:
            code = matches[-1].group(1).strip()

    if code is None:
        code = text.strip()

    # Strip function signature if model re-generated it (HumanEval prompt already has it)
    lines = code.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            body_start = i + 1
            break
    if body_start > 0 and body_start < len(lines):
        code = "\n".join(lines[body_start:])

    return code


_UNICODE_TO_LATEX = {
    "∞": "\\infty",
    "π": "\\pi",
    "·": "\\cdot",
    "×": "\\times",
    "≤": "\\leq",
    "≥": "\\geq",
    "≠": "\\neq",
    "±": "\\pm",
    "∓": "\\mp",
    "√": "\\sqrt",
    "∈": "\\in",
    "∪": "\\cup",
    "∩": "\\cap",
    "⊂": "\\subset",
    "⊃": "\\supset",
    "∅": "\\emptyset",
    "α": "\\alpha",
    "β": "\\beta",
    "γ": "\\gamma",
    "δ": "\\delta",
    "θ": "\\theta",
    "λ": "\\lambda",
    "μ": "\\mu",
    "σ": "\\sigma",
    "ω": "\\omega",
    "°": "^\\circ",
}


def _normalize_math_str(s: str) -> str:
    """Normalize a math answer string for comparison."""
    s = s.strip().strip("$")
    # Only strip trailing dot (not leading — ".5" must stay)
    s = s.rstrip(".")
    # Unicode → LaTeX
    for uchar, latex in _UNICODE_TO_LATEX.items():
        s = s.replace(uchar, latex)
    # Remove LaTeX dollar sign \$, spacing commands \! \, \; \:, and \%
    s = s.replace("\\$", "")
    s = s.replace("\\%", "%")
    s = re.sub(r"\\[!,;:]", "", s)
    # \dfrac/\tfrac → \frac
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    # \frac shorthand normalization (all combos of missing braces):
    # \frac56 → \frac{5}{6}, \frac5{6} → \frac{5}{6}, \frac{5}6 → \frac{5}{6}
    s = re.sub(r"\\frac\s*(\w)\s*(\w)", r"\\frac{\1}{\2}", s)          # \frac56
    s = re.sub(r"\\frac\s*(\w)\s*(\{[^}]+\})", r"\\frac{\1}\2", s)    # \frac5{6}
    s = re.sub(r"\\frac\s*(\{[^}]+\})\s*(\w)", r"\\frac\1{\2}", s)    # \frac{5}6
    # \sqrt7 or \sqrt X (single-char without braces) → \sqrt{X}
    s = re.sub(r"\\sqrt\s*([^{\s])", r"\\sqrt{\1}", s)
    # Remove \text{}, \mathrm{}, \mathit{}, \textbf{} wrappers
    s = re.sub(r"\\(?:text|mathrm|mathit|textbf)\s*\{([^}]*)\}", r"\1", s)
    # Strip trailing unit/text words (2+ chars, e.g. "575 students", "2500 square feet")
    # Does NOT strip single letters (i, r, x — may be math variables)
    s = re.sub(r"\s+[a-zA-Z]{2,}[\sa-zA-Z.]*$", "", s)
    # Remove \left \right
    s = re.sub(r"\\(?:left|right)\s*", "", s)
    # Remove "x \in" prefix (e.g. "x \in [-2,7]" → "[-2,7]")
    s = re.sub(r"^[a-zA-Z]\s*\\in\s*", "", s)
    # Remove spaces
    s = s.replace(" ", "")
    # Strip variable/function assignment prefix: "x=-7" → "-7", "k(x)=-1" → "-1"
    s = re.sub(r"^[a-zA-Z]+\([a-zA-Z]\)\s*=\s*", "", s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    return s


def _latex_frac_to_float(s: str) -> float | None:
    r"""Try to convert \frac{a}{b} to a float."""
    m = re.fullmatch(r"\\frac\{([^}]+)\}\{([^}]+)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    # Negative fraction: -\frac{a}{b}
    m = re.fullmatch(r"-\\frac\{([^}]+)\}\{([^}]+)\}", s)
    if m:
        try:
            return -float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def _latex_to_sympy_str(s: str) -> str:
    """Convert a LaTeX math expression to a string sympy can parse."""
    s = _normalize_math_str(s)
    # \frac{a}{b} → ((a)/(b))
    while "\\frac{" in s:
        idx = s.index("\\frac{")
        # Find the two brace groups
        start1 = s.index("{", idx)
        depth, i = 0, start1
        for i in range(start1, len(s)):
            if s[i] == "{": depth += 1
            elif s[i] == "}": depth -= 1
            if depth == 0: break
        end1 = i
        num = s[start1 + 1:end1]
        start2 = s.index("{", end1 + 1) if end1 + 1 < len(s) and "{" in s[end1 + 1:] else -1
        if start2 == -1:
            break
        depth, i = 0, start2
        for i in range(start2, len(s)):
            if s[i] == "{": depth += 1
            elif s[i] == "}": depth -= 1
            if depth == 0: break
        end2 = i
        den = s[start2 + 1:end2]
        s = s[:idx] + f"(({num})/({den}))" + s[end2 + 1:]
    # \sqrt{a} → sqrt(a)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # Remove remaining backslashes
    s = s.replace("\\", "")
    # ^ → **
    s = s.replace("^", "**")
    # {, } → (, )
    s = s.replace("{", "(").replace("}", ")")
    # Insert implicit multiplication (preserve function names like sqrt, log, etc.)
    # digit followed by letter or '(': 2x → 2*x, 2( → 2*(
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)
    # ')' followed by '(', digit, or letter: )( → )*(, )2 → )*2, )x → )*x
    s = re.sub(r"(\))([a-zA-Z0-9(])", r"\1*\2", s)
    # DO NOT insert * between letters — that would break sqrt, log, sin, etc.
    return s


def _try_parse_sympy(expr_str: str):
    """Try to parse a math expression string with sympy. Returns None on failure."""
    try:
        from sympy.parsing.latex import parse_latex
        result = parse_latex(expr_str)
        if result is not None:
            return result
    except Exception:
        pass
    try:
        from sympy import sympify
        cleaned = _latex_to_sympy_str(expr_str)
        return sympify(cleaned, rational=True)
    except Exception:
        return None


def _is_numeric_equal(a: str, b: str) -> bool:
    """Check if two strings represent the same number.

    Handles: .5 vs 0.5, 1/2 vs 0.5, \\frac{15}{2} vs 7.5, etc.
    """
    def _to_float(s: str) -> float | None:
        # Try \frac{a}{b}
        v = _latex_frac_to_float(s)
        if v is not None:
            return v
        # Try plain fraction a/b
        if "/" in s and "\\" not in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass
        # Try plain float
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None

    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-9
    return False


def _parse_interval(s: str) -> tuple | None:
    """Try to parse an interval like (1,4.5) or [-2,7] or (-inf, inf).

    Returns (left_bracket, left_val, right_val, right_bracket) or None.
    """
    s = _normalize_math_str(s)
    m = re.fullmatch(r"([\[\(])([^,]+),([^,]+)([\]\)])", s)
    if not m:
        return None
    lb, left, right, rb = m.group(1), m.group(2), m.group(3), m.group(4)
    return (lb, left.strip(), right.strip(), rb)


def _intervals_equal(a: str, b: str) -> bool:
    """Check if two interval expressions are equivalent."""
    ia, ib = _parse_interval(a), _parse_interval(b)
    if ia is None or ib is None:
        return False
    # Brackets must match
    if ia[0] != ib[0] or ia[3] != ib[3]:
        return False
    # Compare endpoints (may be numeric or symbolic)
    for va, vb in [(ia[1], ib[1]), (ia[2], ib[2])]:
        if _normalize_math_str(va) == _normalize_math_str(vb):
            continue
        if _is_numeric_equal(va, vb):
            continue
        sa, sb = _try_parse_sympy(va), _try_parse_sympy(vb)
        if sa is not None and sb is not None:
            try:
                from sympy import simplify
                if simplify(sa - sb) == 0:
                    continue
            except Exception:
                pass
        return False
    return True


def math_is_equivalent(pred: str, gold: str, raw_text: str | None = None) -> bool:
    """Check if two math answers are equivalent.

    Tries in order:
    1. Normalized string equality
    2. Numeric equality (handles .5 vs 1/2 vs \\frac{15}{2} vs 7.5)
    3. Interval comparison (handles (1, \\frac{9}{2}) vs (1, 4.5))
    4. math-verify on extracted answers
    5. math-verify on raw generated text (fallback when extraction fails)
    6. Sympy symbolic equality (handles \\sqrt{x} vs x^{1/2})

    Args:
        pred: extracted prediction string
        gold: gold answer string
        raw_text: optional raw generated text for math_verify fallback
    """
    # 1. Normalized string match
    np_, ng = _normalize_math_str(pred), _normalize_math_str(gold)
    if np_ == ng:
        return True

    # 2. Numeric equality
    if _is_numeric_equal(np_, ng):
        return True

    # 3. Interval comparison
    if _intervals_equal(pred, gold):
        return True

    # 3b. Union of intervals: split on \cup and compare pairwise
    cup_pat = r"\\cup"
    pred_parts = re.split(cup_pat, np_)
    gold_parts = re.split(cup_pat, ng)
    if len(pred_parts) == len(gold_parts) and len(pred_parts) > 1:
        if all(_intervals_equal(p.strip(), g.strip()) for p, g in zip(pred_parts, gold_parts)):
            return True

    # 4. math-verify (robust symbolic comparison with pre-normalized input)
    # Skip for very long strings to avoid parse timeouts on degenerate outputs
    if len(np_) <= 500 and len(ng) <= 500:
        try:
            from math_verify import parse as mv_parse, verify as mv_verify
            g = mv_parse(f"${np_}$")
            p = mv_parse(f"${ng}$")
            if mv_verify(g, p):
                return True
        except Exception:
            pass

    # 5. math-verify on raw generated text (when extraction may have failed)
    if raw_text and len(raw_text) <= 50000:
        try:
            from math_verify import parse as mv_parse, verify as mv_verify
            g = mv_parse(gold)
            a = mv_parse(raw_text)
            if mv_verify(g, a):
                return True
        except Exception:
            pass

    # 6. Sympy symbolic equality (fallback)
    sp = _try_parse_sympy(pred)
    sg = _try_parse_sympy(gold)
    if sp is not None and sg is not None:
        try:
            from sympy import simplify
            diff = simplify(sp - sg)
            if diff == 0 or diff.is_zero:
                return True
        except Exception:
            pass

    return False


# Tasks that should use math_is_equivalent for comparison
MATH_EQUIVALENT_TASKS = {
    "math", "math500", "minerva_math", "competition_math",
    "aime2025", "aime2026", "amc23",
}


# Registry of extractors
EXTRACTORS = {
    "gsm8k": _extract_gsm8k,
    "arc": _extract_arc,
    "arc_easy": _extract_arc,
    "arc_challenge": _extract_arc,
    "medqa": _extract_arc,  # also multiple choice
    "aime2025": _extract_math,
    "aime2026": _extract_math,
    "amc23": _extract_math,
    "gpqa_diamond": _extract_math,
    "medqa": _extract_math,
    "competition_math": _extract_competition_math,
    "math": _extract_math,
    "math500": _extract_math,
    "minerva_math": _extract_math,
    "humaneval": _extract_humaneval,
}
