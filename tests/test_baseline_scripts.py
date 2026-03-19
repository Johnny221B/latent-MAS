import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.agent import Agent


def test_extract_last_json_object_from_text():
    from src.cli.run_baseline_paper_latentmas import extract_last_json_object

    text = "line1\n{\"accuracy\": 0.5, \"method\": \"latent_mas\"}\n"
    result = extract_last_json_object(text)
    assert result == {"accuracy": 0.5, "method": "latent_mas"}


def test_normalize_result_converts_fractional_accuracy_to_percent():
    from src.cli.run_baseline_paper_latentmas import normalize_result

    result = normalize_result({"accuracy": 0.0625, "correct": 1})
    assert result["accuracy_raw"] == 0.0625
    assert result["accuracy"] == 6.25


def test_default_output_path_uses_model_and_sample_count():
    from src.cli.run_baseline_single_model import build_default_output_path

    path = build_default_output_path(
        output_dir=Path("outputs"),
        model_name="Qwen/Qwen3-8B",
        max_samples=16,
    )
    assert path.name == "single_model_qwen3-8b_16.json"


def test_build_qwen_chat_messages_uses_user_only_prompt():
    from src.cli.run_baseline_single_model import build_qwen_chat_messages

    assert build_qwen_chat_messages("What is 2+2?") == [
        {"role": "user", "content": "What is 2+2?"},
    ]


def test_build_chat_payload_uses_openai_chat_api_shape():
    from src.cli.run_baseline_single_model import build_chat_payload

    payload = build_chat_payload(
        model_name="Qwen/Qwen3-8B",
        question="What is 2+2?",
        max_new_tokens=128,
        do_sample=False,
    )

    assert payload["model"] == "Qwen/Qwen3-8B"
    assert payload["messages"] == [{"role": "user", "content": "What is 2+2?"}]
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.0


def test_evaluate_one_keeps_full_generated_text(monkeypatch):
    from src.cli import run_baseline_single_model as single_model

    long_text = "x" * 800

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(single_model.requests, "Session", lambda: DummySession())
    monkeypatch.setattr(
        single_model,
        "request_completion",
        lambda session, base_url, api_key, payload: {
            "choices": [{"message": {"content": long_text}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 123},
        },
    )
    monkeypatch.setattr(single_model, "extract_answer", lambda text, task_type: "3")

    result = single_model.evaluate_one(
        index=0,
        sample={"question": "q", "answer": "3"},
        model_name="Qwen/Qwen3-8B",
        max_new_tokens=2048,
        do_sample=False,
        base_url="http://127.0.0.1:8000",
        api_key="EMPTY",
    )

    assert result["generated_text"] == long_text
    assert result["generated_text_preview"] == long_text[:500]


def test_run_single_model_baseline_reports_normal_stop_count(monkeypatch):
    from src.cli import run_baseline_single_model as single_model

    monkeypatch.setattr(
        single_model,
        "create_dataset",
        lambda task, split, max_samples: [
            {"question": "q1", "answer": "1"},
            {"question": "q2", "answer": "2"},
        ],
    )
    monkeypatch.setattr(
        single_model,
        "evaluate_one",
        lambda **kwargs: {
            "index": kwargs["index"],
            "question": kwargs["sample"]["question"],
            "gold": kwargs["sample"]["answer"],
            "prediction": kwargs["sample"]["answer"],
            "generated_text": "ok",
            "generated_text_preview": "ok",
            "generation": {"finish_reason": "stop" if kwargs["index"] == 0 else "length"},
            "correct": True,
        },
    )

    result = single_model.run_single_model_baseline(
        model_name="Qwen/Qwen3-8B",
        max_samples=2,
        max_new_tokens=128,
        do_sample=False,
        worker_count=2,
        base_url="http://127.0.0.1:8000",
        api_key="EMPTY",
    )

    assert result["metrics"]["normal_stop_count"] == 1
    assert result["metrics"]["normal_stop_rate"] == 50.0


def test_infer_finish_reason_eos():
    assert Agent._infer_finish_reason([10, 11, 2], eos_token_id=2, max_new_tokens=8) == "eos"


def test_infer_finish_reason_max_new_tokens():
    assert Agent._infer_finish_reason([10, 11, 12, 13], eos_token_id=2, max_new_tokens=4) == "max_new_tokens"


def test_build_chat_prompt_text_uses_system_and_user_messages():
    class DummyTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            self.calls.append(
                {
                    "messages": messages,
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "enable_thinking": enable_thinking,
                }
            )
            return "agent-chat-prompt"

    tokenizer = DummyTokenizer()
    prompt = Agent.build_chat_prompt_text(
        tokenizer=tokenizer,
        question_text="Solve 2+2.",
        system_prompt="You are a careful math solver.",
        enable_thinking=True,
    )

    assert prompt == "agent-chat-prompt"
    assert tokenizer.calls == [
        {
            "messages": [
                {"role": "system", "content": "You are a careful math solver."},
                {"role": "user", "content": "Solve 2+2."},
            ],
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
    ]
