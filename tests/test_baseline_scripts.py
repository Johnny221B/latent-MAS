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


def test_build_qwen_chat_text_uses_official_chat_template():
    from src.cli.run_baseline_single_model import build_qwen_chat_text

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
            return "formatted-prompt"

    tokenizer = DummyTokenizer()
    text = build_qwen_chat_text(tokenizer, "What is 2+2?")

    assert text == "formatted-prompt"
    assert tokenizer.calls == [
        {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
    ]


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
