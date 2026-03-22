"""GSM8K dataset helpers and config."""


def _load_hf_dataset(dataset_name: str, subset: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, subset, split=split)


def _extract_gsm8k_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def build_task_configs() -> dict:
    return {
        "gsm8k": {
            "loader": lambda split: _load_hf_dataset("openai/gsm8k", "main", split),
            "question_id_field": "id",
            "question_field": "question",
            "answer_field": "answer",
            "answer_extractor": _extract_gsm8k_answer,
        }
    }
