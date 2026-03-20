"""
Unified dataset interface for loading and preprocessing benchmarks.

Each task provides (question, answer) pairs.
The dataset handles tokenization and label preparation.

Extensibility:
  - Add new tasks by adding entries to TASK_CONFIGS.
  - Each config specifies the HF dataset name, split, and field mappings.
"""

import torch
from torch.utils.data import Dataset


class MultiAgentDataset(Dataset):
    """Dataset wrapper that provides (question, answer) pairs for training."""

    def __init__(
        self,
        task: str = "gsm8k",
        split: str = "train",
        max_samples: int | None = None,
    ):
        """
        Args:
            task: task name (key in TASK_CONFIGS)
            split: dataset split
            max_samples: limit number of samples (for debugging)
        """
        from datasets import load_dataset

        task_config = TASK_CONFIGS.get(task)
        if task_config is None:
            raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")

        # Load from HuggingFace datasets
        raw = load_dataset(
            task_config["dataset"],
            task_config.get("subset", None),
            split=split,
        )

        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.data = raw
        self.question_field = task_config["question_field"]
        self.answer_field = task_config["answer_field"]
        self.answer_extractor = task_config.get("answer_extractor", None)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        question = item[self.question_field]

        # Extract clean answer
        raw_answer = item[self.answer_field]
        if self.answer_extractor is not None:
            answer = self.answer_extractor(raw_answer)
        else:
            answer = str(raw_answer)

        return {
            "question": question,
            "answer": answer,
        }


def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numeric answer from GSM8K's answer field."""
    # GSM8K answers are formatted as "explanation\n#### <number>"
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def _extract_arc_answer(answer_key: str) -> str:
    """ARC answers are just the letter key (A/B/C/D)."""
    return str(answer_key).strip()

def _extract_math_answer(solution_text: str) -> str:
    """Extract the boxed answer from MATH dataset solutions.
    
    MATH answers are formatted as \\boxed{answer}.
    """
    import re
    # Look for \boxed{...}
    match = re.search(r'\\boxed\{([^}]*)\}', solution_text)
    if match:
        return match.group(1).strip()
    # Fallback: last line
    return solution_text.strip().split('\n')[-1].strip()

# ── Task Registry ──
TASK_CONFIGS = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "subset": "main",
        "question_field": "question",
        "answer_field": "answer",
        "answer_extractor": _extract_gsm8k_answer,
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "question_field": "question",
        "answer_field": "answerKey",
        "answer_extractor": _extract_arc_answer,
    },
    "arc_challenge": {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "question_field": "question",
        "answer_field": "answerKey",
        "answer_extractor": _extract_arc_answer,
    },
    "math": {
        "dataset": "hendrycks/competition_math",
        "subset": None,
        "question_field": "problem",
        "answer_field": "solution",
        "answer_extractor": _extract_math_answer,
    },
}

def build_labels(
    question_len: int,
    answer_ids: torch.LongTensor,
    ignore_index: int = -100,
) -> torch.LongTensor:
    """Construct labels aligned with [question ; answer] logits.

    Terminal agent's forward_for_loss returns logits for [question ; answer].
    We want loss only on answer positions:
      labels = [-100, -100, ..., -100, answer_token_0, answer_token_1, ...]
               |---- question_len ----||---------- answer_len ----------|

    Args:
        question_len: number of question tokens
        answer_ids: [B, answer_len] ground truth answer tokens
        ignore_index: mask value for non-supervised positions

    Returns:
        labels: [B, question_len + answer_len]
    """
    B, answer_len = answer_ids.shape
    total_len = question_len + answer_len

    labels = torch.full(
        (B, total_len), ignore_index,
        dtype=answer_ids.dtype, device=answer_ids.device,
    )
    # Place answer tokens after question positions
    labels[:, question_len:] = answer_ids

    return labels

def create_dataset(task: str, split: str = "train", max_samples: int | None = None) -> MultiAgentDataset:
    """Factory function to create a dataset.

    Args:
        task: task name
        split: dataset split
        max_samples: limit samples (for debugging)

    Returns:
        MultiAgentDataset instance
    """
    return MultiAgentDataset(task=task, split=split, max_samples=max_samples)
