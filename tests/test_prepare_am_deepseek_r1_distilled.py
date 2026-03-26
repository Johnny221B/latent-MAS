"""Tests for the AM DeepSeek R1 local prepare script."""

import importlib.util
from pathlib import Path


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "scripts" / "prepare_am_deepseek_r1_distilled.py"
    spec = importlib.util.spec_from_file_location("prepare_am_deepseek_r1_distilled", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_row_keeps_full_assistant_output_and_metadata():
    module = _load_prepare_module()

    row = {
        "messages": [
            {"role": "user", "content": "Solve 5+5."},
            {"role": "assistant", "content": "<think>5+5=10</think><answer>10</answer>"},
        ],
        "source": "subset-src",
        "reference_answer": "10",
        "test_case": "basic",
        "info": {
            "think_content": "5+5=10",
            "answer_content": "10",
        },
    }

    normalized = module.normalize_row(row, subset="am_0.9M")

    assert normalized["question"] == "Solve 5+5."
    assert normalized["answer"] == "<think>5+5=10</think><answer>10</answer>"
    assert normalized["subset"] == "am_0.9M"
    assert normalized["question_id"].startswith("am-r1-")
