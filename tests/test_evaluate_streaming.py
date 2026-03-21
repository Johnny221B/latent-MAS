import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_streaming_eval_writes_single_refreshing_json(tmp_path: Path):
    from src.cli.evaluate import write_eval_snapshot

    eval_path = tmp_path / "eval_results.json"

    write_eval_snapshot(
        eval_path=eval_path,
        method="ours_trained_multi_agent",
        task="gsm8k",
        correct=1,
        total=2,
        time_seconds=3.5,
        parameters={"max_samples": 2, "generation_max_new_tokens": 2048},
        world_size=2,
        samples=[
            {
                "question": "q1",
                "gold": "1",
                "prediction": "1",
                "correct": True,
            },
            {
                "question": "q2",
                "gold": "2",
                "prediction": "3",
                "correct": False,
            },
        ],
    )

    payload = json.loads(eval_path.read_text())
    assert payload["method"] == "ours_trained_multi_agent"
    assert payload["metrics"]["correct"] == 1
    assert payload["metrics"]["total"] == 2
    assert payload["metrics"]["accuracy"] == 50.0
    assert len(payload["samples"]) == 2
    assert payload["samples"][0]["question"] == "q1"
    assert payload["samples"][1]["question"] == "q2"


def test_build_agent_sample_log_preserves_agent_summaries():
    from src.cli.evaluate import build_agent_sample_log

    payload = build_agent_sample_log(
        question_id="q-1",
        question="q",
        gold="1",
        prediction="1",
        generation={"finish_reason": "eos", "generated_text": "answer"},
        correct=True,
        agent_logs=[{"agent_id": 0, "output_type": "latent"}],
    )

    assert payload["question_id"] == "q-1"
    assert payload["question"] == "q"
    assert payload["agents"][0]["agent_id"] == 0
    assert payload["generation"]["finish_reason"] == "eos"
    assert payload["generation"]["generated_text"] == "answer"
    assert "generated_text" not in payload


def test_streaming_eval_sample_shape_uses_generation_text_only(tmp_path: Path):
    from src.cli.evaluate import write_eval_snapshot

    eval_path = tmp_path / "eval_results.json"
    write_eval_snapshot(
        eval_path=eval_path,
        method="ours_trained_multi_agent",
        task="gsm8k",
        correct=1,
        total=1,
        time_seconds=1.0,
        parameters={"worker": 2},
        world_size=2,
        samples=[
            {
                "question": "q1",
                "gold": "1",
                "prediction": "1",
                "generation": {
                    "generated_text": "full answer text",
                    "finish_reason": "eos",
                    "generated_token_count": 3,
                    "stopped_early": True,
                },
                "correct": True,
            },
        ],
    )

    payload = json.loads(eval_path.read_text())
    sample = payload["samples"][0]
    assert "generated_text" not in sample
    assert sample["generation"]["generated_text"] == "full answer text"


def test_evaluate_loaded_system_writes_results_without_checkpoint(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    class DummyBaseModel:
        def tokenize(self, texts, max_length=2048, add_special_tokens=True):
            batch = len(texts)
            return {
                "input_ids": torch.ones(batch, 3, dtype=torch.long),
                "attention_mask": torch.ones(batch, 3, dtype=torch.long),
            }

    class DummySystem:
        def __init__(self):
            self.base_model = DummyBaseModel()

        def eval(self):
            return self

        def __call__(
            self,
            task_token_ids,
            task_attention_mask=None,
            max_new_tokens=0,
            inference_mode="chat_with_prefix",
            use_terminal_prefix=True,
            do_sample=False,
            collect_agent_logs=False,
        ):
            batch = task_token_ids.shape[0]
            return {
                "generated_text": ["1"] * batch,
                "generation": {
                    "generated_text": ["1"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [1] * batch,
                    "stopped_early": [True] * batch,
                    "inference_mode": [inference_mode] * batch,
                    "used_upstream_prefix": [use_terminal_prefix] * batch,
                },
                "agent_logs": [],
            }

    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda task, split, max_samples=None: [
            {"question_id": "q-1", "question": "q1", "answer": "1"},
            {"question_id": "q-2", "question": "q2", "answer": "1"},
        ],
    )
    monkeypatch.setattr(evaluate_module, "extract_answer", lambda text, task_type: text.strip())

    result = evaluate_module.evaluate_loaded_system(
        system=DummySystem(),
        config={
            "training": {"task": "gsm8k", "max_seq_len": 32},
        },
        config_path="dummy-config.yaml",
        output_dir=tmp_path,
        checkpoint_path=None,
        max_samples=2,
        split="test",
        max_new_tokens=8,
        inference_mode="chat_with_prefix",
        use_terminal_prefix=True,
        run_baseline=False,
        do_sample=False,
        write_agent_logs=False,
        worker=None,
        batch_size=2,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        is_dist=False,
    )

    eval_path = tmp_path / "eval_results.json"
    payload = json.loads(eval_path.read_text())
    assert eval_path.exists()
    assert payload["parameters"]["checkpoint_path"] is None
    assert payload["metrics"]["correct"] == 2
    assert result["metrics"]["correct"] == 2


def test_evaluate_loaded_system_accepts_chat_with_text_mode(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    class DummyBaseModel:
        def tokenize(self, texts, max_length=2048, add_special_tokens=True):
            batch = len(texts)
            return {
                "input_ids": torch.ones(batch, 3, dtype=torch.long),
                "attention_mask": torch.ones(batch, 3, dtype=torch.long),
            }

    class DummySystem:
        def __init__(self):
            self.base_model = DummyBaseModel()
            self.last_inference_mode = None

        def eval(self):
            return self

        def __call__(
            self,
            task_token_ids,
            task_attention_mask=None,
            max_new_tokens=0,
            inference_mode="chat_with_prefix",
            use_terminal_prefix=True,
            do_sample=False,
            collect_agent_logs=False,
        ):
            self.last_inference_mode = inference_mode
            batch = task_token_ids.shape[0]
            return {
                "generated_text": ["1"] * batch,
                "generation": {
                    "generated_text": ["1"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [1] * batch,
                    "stopped_early": [True] * batch,
                    "inference_mode": [inference_mode] * batch,
                    "used_upstream_prefix": [False] * batch,
                },
                "agent_logs": [],
            }

    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda task, split, max_samples=None: [{"question_id": "q-1", "question": "q1", "answer": "1"}],
    )
    monkeypatch.setattr(evaluate_module, "extract_answer", lambda text, task_type: text.strip())

    system = DummySystem()
    result = evaluate_module.evaluate_loaded_system(
        system=system,
        config={"training": {"task": "gsm8k", "max_seq_len": 32}},
        config_path="dummy-config.yaml",
        output_dir=tmp_path,
        checkpoint_path=None,
        max_samples=1,
        split="test",
        max_new_tokens=8,
        inference_mode="chat_with_text",
        use_terminal_prefix=True,
        run_baseline=False,
        do_sample=False,
        write_agent_logs=False,
        worker=None,
        batch_size=1,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        is_dist=False,
    )

    assert system.last_inference_mode == "chat_with_text"
    assert result["metrics"]["correct"] == 1


def test_evaluate_loaded_system_writes_question_ids_and_role_logs(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    class DummyBaseModel:
        def tokenize(self, texts, max_length=2048, add_special_tokens=True):
            batch = len(texts)
            return {
                "input_ids": torch.ones(batch, 3, dtype=torch.long),
                "attention_mask": torch.ones(batch, 3, dtype=torch.long),
            }

    class DummySystem:
        def __init__(self):
            self.base_model = DummyBaseModel()

        def eval(self):
            return self

        def __call__(
            self,
            task_token_ids,
            task_attention_mask=None,
            max_new_tokens=0,
            inference_mode="chat_with_prefix",
            use_terminal_prefix=True,
            do_sample=False,
            collect_agent_logs=False,
        ):
            batch = task_token_ids.shape[0]
            return {
                "generated_text": ["1"] * batch,
                "generation": {
                    "generated_text": ["1"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [1] * batch,
                    "stopped_early": [True] * batch,
                    "inference_mode": [inference_mode] * batch,
                    "used_upstream_prefix": [False] * batch,
                },
                "agent_logs": [
                    {
                        "agent_id": 0,
                        "role_name": "reader",
                        "output_type": "text_message",
                        "system_prompt": "read",
                        "received_upstream_texts": False,
                        "upstream_texts": [],
                        "generated_text": "reader output",
                        "generation": {"generated_text": "reader output"},
                    },
                    {
                        "agent_id": 1,
                        "role_name": "solver",
                        "output_type": "text",
                        "system_prompt": "solve",
                        "received_upstream_texts": True,
                        "upstream_texts": ["reader output"],
                        "generated_text": "1",
                        "generation": {"generated_text": "1"},
                    },
                ],
            }

    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda task, split, max_samples=None: [
            {"question_id": "gsm8k-test-7", "question": "q1", "answer": "1"},
        ],
    )
    monkeypatch.setattr(evaluate_module, "extract_answer", lambda text, task_type: text.strip())

    result = evaluate_module.evaluate_loaded_system(
        system=DummySystem(),
        config={"training": {"task": "gsm8k", "max_seq_len": 32}},
        config_path="dummy-config.yaml",
        output_dir=tmp_path,
        checkpoint_path=None,
        max_samples=1,
        split="test",
        max_new_tokens=8,
        inference_mode="chat_with_text",
        use_terminal_prefix=True,
        run_baseline=False,
        do_sample=False,
        write_agent_logs=True,
        worker=None,
        batch_size=1,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        is_dist=False,
    )

    eval_payload = json.loads((tmp_path / "eval_results.json").read_text())
    agent_payload = json.loads((tmp_path / "agent_logs.json").read_text())
    reader_role_payload = json.loads((tmp_path / "agent_log" / "reader.json").read_text())
    solver_role_payload = json.loads((tmp_path / "agent_log" / "solver.json").read_text())

    assert eval_payload["samples"][0]["question_id"] == "gsm8k-test-7"
    assert agent_payload["samples"][0]["question_id"] == "gsm8k-test-7"
    assert reader_role_payload["samples"]["gsm8k-test-7"]["output"]["generated_text"] == "reader output"
    assert solver_role_payload["samples"]["gsm8k-test-7"]["input"]["upstream_texts"] == ["reader output"]
    assert result["paths"]["agent_log_path"].endswith("agent_logs.json")


def test_build_single_question_dataset_generates_stable_question_id():
    from src.cli.evaluate import build_single_question_dataset

    dataset = build_single_question_dataset("What is 2+2?")

    assert len(dataset) == 1
    assert dataset[0]["question"] == "What is 2+2?"
    assert dataset[0]["answer"] == ""
    assert dataset[0]["question_id"].startswith("manual-")


def test_evaluate_uses_manual_question_and_custom_output_dir(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    monkeypatch.setattr(
        evaluate_module,
        "load_config",
        lambda path: {"training": {"task": "gsm8k", "max_seq_len": 32}},
    )
    monkeypatch.setattr(
        evaluate_module,
        "setup_eval_distributed",
        lambda: (torch.device("cpu"), 0, 1, False),
    )
    monkeypatch.setattr(evaluate_module, "cleanup_eval_distributed", lambda: None)
    monkeypatch.setattr(evaluate_module, "extract_answer", lambda text, task_type: text.strip())
    monkeypatch.setattr(evaluate_module.torch, "load", lambda *args, **kwargs: {
        "compressor_state": {},
        "adjacency_state": {},
        "base_model_state": None,
    })

    class DummySystem:
        def __init__(self, config):
            self.config = config
            self.base_model = type(
                "DummyBaseModel",
                (),
                {
                    "tokenizer": type("DummyTokenizer", (), {"pad_token_id": 0, "eos_token_id": 2})(),
                    "tokenize": staticmethod(lambda texts, max_length=2048, add_special_tokens=True: {
                        "input_ids": torch.ones(len(texts), 3, dtype=torch.long),
                        "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
                    }),
                    "model": type("DummyModel", (), {"load_state_dict": staticmethod(lambda state: None)})(),
                },
            )()
            self.compressor = type("DummyCompressor", (), {"load_state_dict": staticmethod(lambda state: None)})()
            self.adjacency = type(
                "DummyAdjacency",
                (),
                {
                    "load_state_dict": staticmethod(lambda state: None),
                    "get_adjacency": staticmethod(lambda: torch.zeros(1, 1)),
                    "get_hard_adjacency": staticmethod(lambda: torch.zeros(1, 1)),
                },
            )()

        def to(self, device):
            return self

        def eval(self):
            return self

        def log_adjacency(self):
            return "Adjacency (1x1):"

        def __call__(
            self,
            task_token_ids,
            task_attention_mask=None,
            max_new_tokens=0,
            inference_mode="chat_with_prefix",
            use_terminal_prefix=True,
            do_sample=False,
            collect_agent_logs=False,
        ):
            batch = task_token_ids.shape[0]
            return {
                "generated_text": ["4"] * batch,
                "generation": {
                    "generated_text": ["4"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [1] * batch,
                    "stopped_early": [True] * batch,
                    "inference_mode": [inference_mode] * batch,
                    "used_upstream_prefix": [use_terminal_prefix] * batch,
                },
                "agent_logs": [
                    {
                        "agent_id": 0,
                        "role_name": "solver",
                        "output_type": "text",
                        "system_prompt": "solve",
                        "received_upstream_texts": False,
                        "upstream_texts": [],
                        "generated_text": "4",
                        "generation": {"generated_text": "4"},
                    }
                ],
            }

    monkeypatch.setattr(evaluate_module, "MultiAgentSystem", DummySystem)
    create_dataset_called = {"value": False}
    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda *args, **kwargs: create_dataset_called.__setitem__("value", True),
    )

    evaluate_module.evaluate(
        config_path="dummy-config.yaml",
        checkpoint_path="dummy-checkpoint.pt",
        max_samples=None,
        split="test",
        max_new_tokens=16,
        inference_mode="chat_with_text",
        use_terminal_prefix=True,
        run_baseline=False,
        do_sample=False,
        write_agent_logs=True,
        worker=1,
        batch_size=1,
        question="What is 2+2?",
        output_dir=tmp_path,
    )

    payload = json.loads((tmp_path / "eval_results.json").read_text())
    role_payload = json.loads((tmp_path / "agent_log" / "solver.json").read_text())
    assert create_dataset_called["value"] is False
    assert payload["samples"][0]["question"] == "What is 2+2?"
    assert payload["samples"][0]["question_id"].startswith("manual-")
    assert role_payload["samples"][payload["samples"][0]["question_id"]]["output"]["generated_text"] == "4"
