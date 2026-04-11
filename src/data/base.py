"""Shared dataset wrapper and label helpers."""

import hashlib

import torch
from torch.utils.data import Dataset


class MultiAgentDataset(Dataset):
    """Dataset wrapper that provides normalized samples for training/eval."""

    def __init__(
        self,
        task: str = "gsm8k",
        split: str = "train",
        max_samples: int | None = None,
        **loader_kwargs,
    ):
        from .factory import get_task_configs

        task_config = get_task_configs().get(task)
        if task_config is None:
            raise ValueError(f"Unknown task: {task}. Available: {list(get_task_configs().keys())}")

        allowed_splits = task_config.get("allowed_splits")
        if allowed_splits is not None and split not in allowed_splits:
            raise ValueError(
                f"Unsupported split for {task}. "
                f"Available: {sorted(allowed_splits)}"
            )

        raw = task_config["loader"](split, **loader_kwargs)
        split_handler = task_config.get("split_handler")
        if split_handler is not None:
            raw = split_handler(raw, split)

        if max_samples is not None:
            n = min(max_samples, len(raw))
            raw = raw.select(range(n)) if hasattr(raw, "select") else raw[:n]

        self.data = raw
        self.question_field = task_config["question_field"]
        self.answer_field = task_config["answer_field"]
        self.question_id_field = task_config["question_id_field"]
        self.answer_extractor = task_config.get("answer_extractor")
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
        question_id = self._resolve_question_id(item, question, idx)

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

    def _resolve_question_id(self, item: dict, question: str, idx: int) -> str:
        question_id_field = getattr(self, "question_id_field", None)
        if question_id_field and question_id_field in item:
            return str(item[question_id_field])

        stable_key = f"{self.__class__.__name__}:{idx}:{question}".encode("utf-8")
        digest = hashlib.sha1(stable_key).hexdigest()[:12]
        return f"sample-{digest}"


def build_labels(
    question_len: int,
    answer_ids: torch.LongTensor,
    ignore_index: int = -100,
) -> torch.LongTensor:
    """Construct labels aligned with [question ; answer] logits."""
    batch_size, answer_len = answer_ids.shape
    total_len = question_len + answer_len

    labels = torch.full(
        (batch_size, total_len),
        ignore_index,
        dtype=answer_ids.dtype,
        device=answer_ids.device,
    )
    labels[:, question_len:] = answer_ids
    return labels
