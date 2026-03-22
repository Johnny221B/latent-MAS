import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_merge_eval_config_overrides_task_and_evaluation_fields(tmp_path: Path):
    from src.cli.evaluate import merge_eval_config

    base_config_path = tmp_path / "train_config.yaml"
    eval_config_path = tmp_path / "eval_config.yaml"
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "agents": ["planner", "solver", "critic"],
                "adjacency_prior": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
                "execution_order": [0, 1, 2],
                "terminal_agent_index": 2,
            }
        ),
        encoding="utf-8",
    )
    base_config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "dummy"},
                "graph": {"config": str(graph_path)},
                "training": {"task": "competition_math", "batch_size": 2},
                "evaluation": {"split": "train", "max_new_tokens": 64},
            }
        ),
        encoding="utf-8",
    )
    eval_config_path.write_text(
        yaml.safe_dump(
            {
                "task": "gsm8k",
                "split": "test",
                "max_new_tokens": 128,
                "batch_size": 1,
            }
        ),
        encoding="utf-8",
    )

    merged = merge_eval_config(base_config_path, eval_config_path)

    assert merged["training"]["task"] == "gsm8k"
    assert merged["evaluation"]["split"] == "test"
    assert merged["evaluation"]["max_new_tokens"] == 128
    assert merged["evaluation"]["batch_size"] == 1


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


def test_evaluate_loaded_system_scores_arc_letter_predictions(tmp_path: Path, monkeypatch):
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
                "generated_text": ["The answer is B"] * batch,
                "generation": {
                    "generated_text": ["The answer is B"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [4] * batch,
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
            {
                "question_id": "arc-test-1",
                "question": "Which planet is known as the Red Planet?\n\nChoices:\nA. Earth\nB. Mars",
                "answer": "B",
            }
        ],
    )

    result = evaluate_module.evaluate_loaded_system(
        system=DummySystem(),
        config={"training": {"task": "arc_easy", "max_seq_len": 32}},
        config_path="dummy-config.yaml",
        output_dir=tmp_path,
        checkpoint_path=None,
        max_samples=1,
        split="test",
        max_new_tokens=8,
        inference_mode="chat_with_prefix",
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

    payload = json.loads((tmp_path / "eval_results.json").read_text())
    assert payload["metrics"]["correct"] == 1
    assert payload["task"] == "arc_easy"
    assert payload["samples"][0]["prediction"] == "B"
    assert result["metrics"]["correct"] == 1


def test_evaluate_loaded_system_writes_humaneval_samples_and_pass_at_k(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    class DummyBaseModel:
        def tokenize(self, texts, max_length=2048, add_special_tokens=True):
            batch = len(texts)
            return {
                "input_ids": torch.ones(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }

    class DummySystem:
        def __init__(self):
            self.base_model = DummyBaseModel()
            self.calls = []

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
            self.calls.append(
                {
                    "batch": task_token_ids.shape[0],
                    "do_sample": do_sample,
                    "inference_mode": inference_mode,
                    "use_terminal_prefix": use_terminal_prefix,
                }
            )
            batch = task_token_ids.shape[0]
            return {
                "generated_text": ["def add_one(x):\n    return x + 1"] * batch,
                "generation": {
                    "generated_text": ["def add_one(x):\n    return x + 1"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [6] * batch,
                    "stopped_early": [True] * batch,
                    "inference_mode": [inference_mode] * batch,
                    "used_upstream_prefix": [use_terminal_prefix] * batch,
                },
                "agent_logs": [],
            }

    humaneval_dataset = [
        {
            "question_id": "HumanEval/0",
            "question": "def add_one(x):\n    ",
            "answer": "return x + 1",
            "task_id": "HumanEval/0",
            "prompt": "def add_one(x):\n    ",
            "canonical_solution": "return x + 1",
            "test": "assert add_one(1) == 2",
            "entry_point": "add_one",
        },
        {
            "question_id": "HumanEval/1",
            "question": "def add_two(x):\n    ",
            "answer": "return x + 2",
            "task_id": "HumanEval/1",
            "prompt": "def add_two(x):\n    ",
            "canonical_solution": "return x + 2",
            "test": "assert add_two(1) == 3",
            "entry_point": "add_two",
        },
    ]

    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda task, split, max_samples=None: humaneval_dataset,
    )
    monkeypatch.setattr(
        evaluate_module,
        "extract_answer",
        lambda text, task_type: text.strip(),
    )
    monkeypatch.setattr(
        evaluate_module,
        "_evaluate_humaneval_samples",
        lambda sample_path, *, num_samples_per_task, pass_at_k, config, problems, output_dir: {
            "metrics": {"pass@1": 0.5, "pass@10": 1.0},
            "results_path": str(sample_path) + "_results.jsonl",
            "problem_path": "unused",
            "pass_at_k": list(pass_at_k),
        },
    )

    system = DummySystem()
    result = evaluate_module.evaluate_loaded_system(
        system=system,
        config={
            "training": {"task": "humaneval", "max_seq_len": 32},
            "evaluation": {
                "num_samples_per_task": 2,
                "pass_at_k": [1, 10],
                "do_sample": True,
                "temperature": 0.8,
            },
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
        do_sample=True,
        write_agent_logs=False,
        worker=None,
        batch_size=1,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        is_dist=False,
    )

    sample_path = tmp_path / "humaneval_samples.jsonl"
    eval_payload = json.loads((tmp_path / "eval_results.json").read_text())

    assert sample_path.exists()
    sample_lines = sample_path.read_text().strip().splitlines()
    assert len(sample_lines) == 4
    first_row = json.loads(sample_lines[0])
    assert first_row["task_id"] == "HumanEval/0"
    assert "completion" in first_row
    assert eval_payload["metrics"]["pass@1"] == 0.5
    assert eval_payload["metrics"]["pass@10"] == 1.0
    assert result["metrics"]["pass@1"] == 0.5
    assert result["paths"]["samples_path"].endswith("humaneval_samples.jsonl")
    assert result["paths"]["results_path"].endswith("_results.jsonl")
    assert system.calls and system.calls[0]["do_sample"] is True


def test_evaluate_loaded_system_humaneval_hard_fails_without_human_eval(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    class DummyBaseModel:
        def tokenize(self, texts, max_length=2048, add_special_tokens=True):
            batch = len(texts)
            return {
                "input_ids": torch.ones(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }

    class DummySystem:
        def __init__(self):
            self.base_model = DummyBaseModel()

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            task_token_ids = kwargs["task_token_ids"]
            return {
                "generated_text": ["pass"] * task_token_ids.shape[0],
                "generation": {
                    "generated_text": ["pass"] * task_token_ids.shape[0],
                    "finish_reason": ["eos"] * task_token_ids.shape[0],
                    "generated_token_count": [1] * task_token_ids.shape[0],
                    "stopped_early": [True] * task_token_ids.shape[0],
                    "inference_mode": [kwargs.get("inference_mode", "chat_with_prefix")] * task_token_ids.shape[0],
                    "used_upstream_prefix": [kwargs.get("use_terminal_prefix", True)] * task_token_ids.shape[0],
                },
                "agent_logs": [],
            }

    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda task, split, max_samples=None: [
            {
                "question_id": "HumanEval/0",
                "question": "def add_one(x):\n    ",
                "answer": "return x + 1",
                "task_id": "HumanEval/0",
                "prompt": "def add_one(x):\n    ",
                "canonical_solution": "return x + 1",
                "test": "assert add_one(1) == 2",
                "entry_point": "add_one",
            }
        ],
    )
    monkeypatch.setattr(evaluate_module, "_import_human_eval_evaluation", lambda: None)

    with pytest.raises(RuntimeError, match="human_eval"):
        evaluate_module.evaluate_loaded_system(
            system=DummySystem(),
            config={
                "training": {"task": "humaneval", "max_seq_len": 32},
                "evaluation": {"num_samples_per_task": 1, "pass_at_k": [1]},
            },
            config_path="dummy-config.yaml",
            output_dir=tmp_path,
            checkpoint_path=None,
            max_samples=1,
            split="test",
            max_new_tokens=8,
            inference_mode="chat_with_prefix",
            use_terminal_prefix=True,
            run_baseline=False,
            do_sample=True,
            write_agent_logs=False,
            worker=None,
            batch_size=1,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            is_dist=False,
        )


def test_evaluate_loaded_system_humaneval_writes_agent_logs(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    class DummyBaseModel:
        def tokenize(self, texts, max_length=2048, add_special_tokens=True):
            batch = len(texts)
            return {
                "input_ids": torch.ones(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
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
                "generated_text": ["return x + 1"] * batch,
                "generation": {
                    "generated_text": ["return x + 1"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [4] * batch,
                    "stopped_early": [True] * batch,
                },
                "agent_logs": [
                    {
                        "agent_id": 0,
                        "role_name": "reader",
                        "output_type": "text_message",
                        "system_prompt": "read",
                        "received_upstream_texts": False,
                        "upstream_texts": [],
                        "generated_text": "inspect signature",
                        "generation": {"generated_text": "inspect signature"},
                    },
                    {
                        "agent_id": 1,
                        "role_name": "solver",
                        "output_type": "text",
                        "system_prompt": "solve",
                        "received_upstream_texts": True,
                        "upstream_texts": ["inspect signature"],
                        "generated_text": "return x + 1",
                        "generation": {"generated_text": "return x + 1"},
                    },
                ] if collect_agent_logs else [],
            }

    humaneval_dataset = [
        {
            "question_id": "HumanEval/0",
            "question": "def add_one(x):\n    ",
            "answer": "return x + 1",
            "task_id": "HumanEval/0",
            "prompt": "def add_one(x):\n    ",
            "canonical_solution": "return x + 1",
            "test": "assert add_one(1) == 2",
            "entry_point": "add_one",
        }
    ]

    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda task, split, max_samples=None: humaneval_dataset,
    )
    monkeypatch.setattr(
        evaluate_module,
        "_evaluate_humaneval_samples",
        lambda sample_path, *, num_samples_per_task, pass_at_k, config, problems, output_dir: {
            "metrics": {"pass@1": 1.0},
            "results_path": str(sample_path) + "_results.jsonl",
            "problem_path": str(output_dir / "humaneval_problems.jsonl"),
            "pass_at_k": list(pass_at_k),
        },
    )

    result = evaluate_module.evaluate_loaded_system(
        system=DummySystem(),
        config={
            "training": {"task": "humaneval", "max_seq_len": 32},
            "evaluation": {"num_samples_per_task": 1, "pass_at_k": [1]},
        },
        config_path="dummy-config.yaml",
        output_dir=tmp_path,
        checkpoint_path=None,
        max_samples=1,
        split="test",
        max_new_tokens=8,
        inference_mode="chat_with_prefix",
        use_terminal_prefix=True,
        run_baseline=False,
        do_sample=True,
        write_agent_logs=True,
        worker=None,
        batch_size=1,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        is_dist=False,
    )

    agent_payload = json.loads((tmp_path / "agent_logs.json").read_text())
    reader_payload = json.loads((tmp_path / "agent_log" / "reader.json").read_text())
    solver_payload = json.loads((tmp_path / "agent_log" / "solver.json").read_text())

    assert agent_payload["samples"][0]["question_id"] == "HumanEval/0"
    assert reader_payload["samples"]["HumanEval/0"]["output"]["generated_text"] == "inspect signature"
    assert solver_payload["samples"]["HumanEval/0"]["input"]["upstream_texts"] == ["inspect signature"]
    assert result["paths"]["agent_log_path"].endswith("agent_logs.json")
    assert result["paths"]["role_agent_log_dir"].endswith("agent_log")


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
    assert "agent_log" not in eval_payload["samples"][0]
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


def test_evaluate_cli_delegates_humaneval_to_loaded_system(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    monkeypatch.setattr(
        evaluate_module,
        "load_config",
        lambda path: {
            "training": {"task": "humaneval", "max_seq_len": 32},
            "evaluation": {"num_samples_per_task": 2, "pass_at_k": [1]},
        },
    )
    monkeypatch.setattr(
        evaluate_module,
        "setup_eval_distributed",
        lambda: (torch.device("cpu"), 0, 1, False),
    )
    monkeypatch.setattr(evaluate_module, "cleanup_eval_distributed", lambda: None)
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

    monkeypatch.setattr(evaluate_module, "MultiAgentSystem", DummySystem)

    called = {}

    def fake_evaluate_loaded_system(**kwargs):
        called["kwargs"] = kwargs
        return {"task": "humaneval"}

    monkeypatch.setattr(evaluate_module, "evaluate_loaded_system", fake_evaluate_loaded_system)

    evaluate_module.evaluate(
        config_path="dummy-config.yaml",
        checkpoint_path="dummy-checkpoint.pt",
        max_samples=3,
        split="test",
        max_new_tokens=64,
        inference_mode="chat_with_prefix",
        use_terminal_prefix=True,
        run_baseline=False,
        do_sample=True,
        write_agent_logs=True,
        worker=2,
        batch_size=1,
        question=None,
        output_dir=tmp_path,
    )

    assert called["kwargs"]["config"]["training"]["task"] == "humaneval"
    assert called["kwargs"]["split"] == "test"
    assert called["kwargs"]["max_new_tokens"] == 64
    assert called["kwargs"]["output_dir"] == tmp_path
