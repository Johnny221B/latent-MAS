"""Tests for dataset question ids."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import MultiAgentDataset


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
