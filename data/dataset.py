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


def _load_hf_dataset(dataset_name: str, subset: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, subset, split=split)


def _apply_humaneval_split(raw, split: str):
    total = len(raw)
    train_cutoff = int(total * 0.6)

    split_to_indices = {
        "train": list(range(0, train_cutoff)),
        "test": list(range(train_cutoff, total)),
    }
    if split not in split_to_indices:
        raise ValueError(
            "Unsupported split for humaneval. "
            "Use 'train' or 'test' for the local 60/40 debug split."
        )

    return raw.select(split_to_indices[split])


def _iter_arc_choices(raw_choices) -> list[tuple[str, str]]:
    if isinstance(raw_choices, dict):
        labels = raw_choices.get("label", ())
        texts = raw_choices.get("text", ())
        return [
            (str(label).strip().upper(), str(text).strip())
            for label, text in zip(labels, texts)
        ]

    if isinstance(raw_choices, list):
        pairs = []
        for choice in raw_choices:
            if not isinstance(choice, dict):
                continue
            label = str(choice.get("label", "")).strip().upper()
            text = str(choice.get("text", "")).strip()
            if label and text:
                pairs.append((label, text))
        return pairs

    return []


def _format_arc_question(item: dict) -> str:
    question = str(item.get("question", "")).strip()
    choice_lines = [
        f"{label}. {text}"
        for label, text in _iter_arc_choices(item.get("choices"))
        if label and text
    ]
    if not choice_lines:
        return question
    return f"{question}\n\nChoices:\n" + "\n".join(choice_lines)


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
        task_config = TASK_CONFIGS.get(task)
        if task_config is None:
            raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")

        split_handler = task_config.get("split_handler")
        allowed_splits = task_config.get("allowed_splits")
        if allowed_splits is not None and split not in allowed_splits:
            raise ValueError(
                f"Unsupported split for {task}. "
                f"Available: {sorted(allowed_splits)}"
            )

        # Load from HuggingFace datasets
        raw = _load_hf_dataset(
            task_config["dataset"],
            task_config.get("subset", None),
            task_config.get("hf_split", split),
        )
        if split_handler is not None:
            raw = split_handler(raw, split)

        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.data = raw
        self.question_field = task_config["question_field"]
        self.answer_field = task_config["answer_field"]
        self.question_id_field = task_config["question_id_field"]
        self.answer_extractor = task_config.get("answer_extractor", None)
        self.question_formatter = task_config.get("question_formatter")
        self.extra_fields = tuple(task_config.get("extra_fields", ()))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        question_formatter = getattr(self, "question_formatter", None)
        if question_formatter is not None:
            question = question_formatter(item)
        else:
            question = item[self.question_field]
        question_id = item[self.question_id_field]

        # Extract clean answer
        raw_answer = item[self.answer_field]
        if self.answer_extractor is not None:
            answer = self.answer_extractor(raw_answer)
        else:
            answer = str(raw_answer)

        return {
            "question_id": str(question_id),
            "question": question,
            "answer": answer,
            **{
                field: item[field]
                for field in getattr(self, "extra_fields", ())
                if field in item
            },
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


# ── Task Registry ──
TASK_CONFIGS = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "subset": "main",
        "question_id_field": "id",
        "question_field": "question",
        "answer_field": "answer",
        "answer_extractor": _extract_gsm8k_answer,
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "question_id_field": "id",
        "question_field": "question",
        "question_formatter": _format_arc_question,
        "answer_field": "answerKey",
        "answer_extractor": _extract_arc_answer,
    },
    "arc_challenge": {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "question_id_field": "id",
        "question_field": "question",
        "question_formatter": _format_arc_question,
        "answer_field": "answerKey",
        "answer_extractor": _extract_arc_answer,
    },
    "humaneval": {
        "dataset": "openai_humaneval",
        "subset": None,
        "hf_split": "test",
        "split_handler": _apply_humaneval_split,
        "allowed_splits": ("train", "test"),
        "question_id_field": "task_id",
        "question_field": "prompt",
        "answer_field": "canonical_solution",
        "extra_fields": ("task_id", "prompt", "canonical_solution", "test", "entry_point"),
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
