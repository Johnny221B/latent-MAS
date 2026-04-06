"""HumanEval dataset helpers and config."""


def _load_hf_dataset(split: str):
    from datasets import load_dataset

    return load_dataset("openai_humaneval", split="test")


def build_task_configs() -> dict:
    return {
        "humaneval": {
            "loader": _load_hf_dataset,
            "allowed_splits": ("test",),
            "question_id_field": "task_id",
            "question_field": "prompt",
            "answer_field": "canonical_solution",
            "extra_fields": ("task_id", "prompt", "canonical_solution", "test", "entry_point"),
        }
    }
