"""Tests for dataset question ids."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import MultiAgentDataset
from src.data import get_task_configs


def test_dataset_factory_registry_contains_supported_tasks():
    task_configs = get_task_configs()

    assert {"gsm8k", "arc_easy", "arc_challenge", "humaneval", "competition_math"} <= set(task_configs)


def test_dataset_returns_question_id_from_raw_record():
    dataset = MultiAgentDataset.__new__(MultiAgentDataset)
    dataset.data = [
        {
            "id": "gsm8k-train-3",
            "question": "What is 1+2?",
            "answer": "explain\n#### 3",
        }
    ]
    dataset.question_field = "question"
    dataset.answer_field = "answer"
    dataset.answer_extractor = lambda text: text.split("####")[-1].strip()
    dataset.question_id_field = "id"

    sample = dataset[0]

    assert sample["question_id"] == "gsm8k-train-3"
    assert sample["question"] == "What is 1+2?"
    assert sample["answer"] == "3"


def test_format_arc_question_renders_labeled_choices():
    from src.data import arc as dataset_module

    formatted = dataset_module._format_arc_question(
        {
            "question": "Which planet is known as the Red Planet?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["Earth", "Mars", "Jupiter", "Venus"],
            },
        }
    )

    assert formatted == (
        "Which planet is known as the Red Planet?\n\n"
        "Choices:\n"
        "A. Earth\n"
        "B. Mars\n"
        "C. Jupiter\n"
        "D. Venus"
    )


def test_arc_dataset_formats_question_with_choices(monkeypatch):
    from src.data import arc as dataset_module

    raw_rows = [
        {
            "id": "arc-easy-train-1",
            "question": "Which planet is known as the Red Planet?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["Earth", "Mars", "Jupiter", "Venus"],
            },
            "answerKey": "B",
        }
    ]

    class DummyRawDataset:
        def __init__(self, rows):
            self.rows = list(rows)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

        def select(self, indices):
            return DummyRawDataset([self.rows[idx] for idx in indices])

    def fake_load_dataset(dataset_name, subset, split):
        assert dataset_name == "allenai/ai2_arc"
        assert subset == "ARC-Easy"
        assert split == "train"
        return DummyRawDataset(raw_rows)

    monkeypatch.setattr(dataset_module, "_load_hf_dataset", fake_load_dataset)

    dataset = MultiAgentDataset(task="arc_easy", split="train")
    sample = dataset[0]

    assert sample["question_id"] == "arc-easy-train-1"
    assert sample["question"].startswith("Which planet is known as the Red Planet?")
    assert "Choices:\nA. Earth\nB. Mars\nC. Jupiter\nD. Venus" in sample["question"]
    assert sample["answer"] == "B"


def test_humaneval_dataset_preserves_code_metadata():
    dataset = MultiAgentDataset.__new__(MultiAgentDataset)
    dataset.data = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def add_one(x):\n    ",
            "canonical_solution": "return x + 1",
            "test": "assert add_one(1) == 2",
            "entry_point": "add_one",
        }
    ]
    dataset.question_field = "prompt"
    dataset.answer_field = "canonical_solution"
    dataset.question_id_field = "task_id"
    dataset.answer_extractor = None
    dataset.extra_fields = ("task_id", "prompt", "canonical_solution", "test", "entry_point")

    sample = dataset[0]

    assert sample["question_id"] == "HumanEval/0"
    assert sample["question"] == "def add_one(x):\n    "
    assert sample["answer"] == "return x + 1"
    assert sample["task_id"] == "HumanEval/0"
    assert sample["prompt"] == "def add_one(x):\n    "
    assert sample["canonical_solution"] == "return x + 1"
    assert sample["test"] == "assert add_one(1) == 2"
    assert sample["entry_point"] == "add_one"


def test_humaneval_dataset_uses_deterministic_60_40_split(monkeypatch):
    from src.data import humaneval as dataset_module

    raw_rows = [
        {
            "task_id": f"HumanEval/{idx}",
            "prompt": f"def fn_{idx}():\n    ",
            "canonical_solution": f"return {idx}",
            "test": f"assert fn_{idx}() == {idx}",
            "entry_point": f"fn_{idx}",
        }
        for idx in range(10)
    ]

    class DummyRawDataset:
        def __init__(self, rows):
            self.rows = list(rows)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

        def select(self, indices):
            return DummyRawDataset([self.rows[idx] for idx in indices])

    def fake_load_dataset(dataset_name, subset, split):
        assert dataset_name == "openai_humaneval"
        assert split == "test"
        return DummyRawDataset(raw_rows)

    monkeypatch.setattr(dataset_module, "_load_hf_dataset", fake_load_dataset)

    train_dataset = MultiAgentDataset(task="humaneval", split="train")
    test_dataset = MultiAgentDataset(task="humaneval", split="test")

    assert len(train_dataset) == 6
    assert len(test_dataset) == 4
    assert train_dataset[0]["question_id"] == "HumanEval/0"
    assert train_dataset[5]["question_id"] == "HumanEval/5"
    assert test_dataset[0]["question_id"] == "HumanEval/6"
    assert test_dataset[3]["question_id"] == "HumanEval/9"


def test_humaneval_dataset_rejects_validation_split(monkeypatch):
    from src.data import humaneval as dataset_module

    monkeypatch.setattr(
        dataset_module,
        "_load_hf_dataset",
        lambda dataset_name, subset, split: pytest.fail("load_dataset should not be called"),
    )

    with pytest.raises(ValueError, match="Unsupported split"):
        MultiAgentDataset(task="humaneval", split="validation")


def test_competition_math_dataset_extracts_boxed_answer(monkeypatch):
    from src.data import competition_math as dataset_module

    raw_rows = [
        {
            "problem": "Compute 6 * 7.",
            "solution": "We get 42, so the final answer is \\boxed{42}.",
            "level": "Level 1",
            "type": "algebra",
        }
    ]

    class DummyRawDataset:
        def __init__(self, rows):
            self.rows = list(rows)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

        def select(self, indices):
            return DummyRawDataset([self.rows[idx] for idx in indices])

    monkeypatch.setattr(
        dataset_module,
        "_load_hf_dataset",
        lambda dataset_name, subset, split: DummyRawDataset(raw_rows),
    )

    dataset = MultiAgentDataset(task="competition_math", split="train")
    sample = dataset[0]

    assert sample["question_id"] == "Compute 6 * 7."
    assert sample["question"] == "Compute 6 * 7."
    assert sample["answer"] == "42"
    assert sample["level"] == "Level 1"
    assert sample["type"] == "algebra"
