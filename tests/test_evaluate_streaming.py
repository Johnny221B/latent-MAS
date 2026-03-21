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
        question="q",
        gold="1",
        prediction="1",
        generation={"finish_reason": "eos", "generated_text": "answer"},
        correct=True,
        agent_logs=[{"agent_id": 0, "output_type": "latent"}],
    )

    assert payload["question"] == "q"
    assert payload["agents"][0]["agent_id"] == 0
    assert payload["generation"]["finish_reason"] == "eos"
    assert payload["generation"]["generated_text"] == "answer"
    assert "generated_text" not in payload


def test_is_suspected_gibberish_flags_known_bad_patterns():
    from src.cli.evaluate import is_suspected_gibberish

    assert is_suspected_gibberish("ДДДДДДДДДД TableTableTable ; ; ;") is True
    assert is_suspected_gibberish("Reasoning... therefore #### 42") is False


def test_print_sample_preview_marks_warning_for_gibberish(capsys):
    from src.cli.evaluate import print_sample_preview

    flagged = print_sample_preview(
        sample_number=1,
        question="What is 2+2?",
        gold="4",
        prediction="",
        generation={
            "generated_text": "ДДДДДДДДДД TableTableTable",
            "finish_reason": "max_new_tokens",
            "generated_token_count": 12000,
        },
    )

    captured = capsys.readouterr().out
    assert flagged is True
    assert "Sample 1 [WARN]" in captured
    assert "Finish: max_new_tokens | Tokens: 12000" in captured


def test_run_preflight_preview_skips_when_preview_limit_is_zero(monkeypatch):
    from src.cli import evaluate as evaluate_module

    called = {"run_system_batch": 0}

    def fake_run_system_batch(*args, **kwargs):
        called["run_system_batch"] += 1
        return [], None

    monkeypatch.setattr(evaluate_module, "run_system_batch", fake_run_system_batch)
    monkeypatch.setattr(evaluate_module, "gather_sharded_objects", lambda obj, rank, world_size: [obj])

    preview_printed, suspected_gibberish_count = evaluate_module.run_preflight_preview(
        system=object(),
        config={"training": {"task": "gsm8k"}},
        dataset=[{"question": "q1", "answer": "1"}],
        task="gsm8k",
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        generation_max_new_tokens=32,
        inference_mode="chat_with_prefix",
        use_terminal_prefix=True,
        communication_mode="latent_prefix",
        text_message_edge_threshold=0.5,
        text_message_max_new_tokens=32,
        do_sample=False,
        write_agent_logs=False,
        preview_limit=0,
    )

    assert preview_printed == 0
    assert suspected_gibberish_count == 0
    assert called["run_system_batch"] == 0


def test_select_agent_logs_for_sample_slices_batch_generation_metadata():
    from src.cli.evaluate import select_agent_logs_for_sample

    sample_logs = select_agent_logs_for_sample(
        [
            {
                "agent_id": 0,
                "output_type": "latent",
                "hidden_trajectory": {"shape": [2, 25, 4096], "norm": 1.0},
            },
            {
                "agent_id": 4,
                "output_type": "text",
                "generated_text": ["first", "second"],
                "generation": {
                    "finish_reason": ["eos", "max_new_tokens"],
                    "generated_token_count": [3, 8],
                    "stopped_early": [True, False],
                },
            }
        ],
        index=1,
    )

    assert sample_logs[0]["hidden_trajectory"]["shape"] == [2, 25, 4096]
    assert sample_logs[1]["generated_text"] == "second"
    assert sample_logs[1]["generation"]["finish_reason"] == "max_new_tokens"
    assert sample_logs[1]["generation"]["generated_token_count"] == 8
    assert sample_logs[1]["generation"]["stopped_early"] is False


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
            {"question": "q1", "answer": "1"},
            {"question": "q2", "answer": "1"},
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


def test_evaluate_loaded_system_runs_preflight_preview_before_bulk_eval(tmp_path: Path, monkeypatch):
    from src.cli import evaluate as evaluate_module

    call_sizes = []

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
            call_sizes.append(batch)
            return {
                "generated_text": ["1"] * batch,
                "generation": {
                    "generated_text": ["1"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [1] * batch,
                    "stopped_early": [True] * batch,
                },
                "agent_logs": [],
            }

    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda task, split, max_samples=None: [
            {"question": f"q{i}", "answer": "1"}
            for i in range(4)
        ],
    )
    monkeypatch.setattr(evaluate_module, "extract_answer", lambda text, task_type: text.strip())

    evaluate_module.evaluate_loaded_system(
        system=DummySystem(),
        config={"training": {"task": "gsm8k", "max_seq_len": 32}},
        config_path="dummy-config.yaml",
        output_dir=tmp_path,
        checkpoint_path=None,
        max_samples=4,
        split="test",
        max_new_tokens=8,
        inference_mode="chat_with_prefix",
        use_terminal_prefix=True,
        run_baseline=False,
        do_sample=False,
        write_agent_logs=False,
        worker=None,
        batch_size=4,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        is_dist=False,
    )

    assert call_sizes[:4] == [1, 1, 1, 1]
    assert call_sizes[4:] == [4]


def test_evaluate_loaded_system_writes_progress_log(tmp_path: Path, monkeypatch):
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
                },
                "agent_logs": [],
            }

    monkeypatch.setattr(
        evaluate_module,
        "create_dataset",
        lambda task, split, max_samples=None: [{"question": "q1", "answer": "1"}],
    )
    monkeypatch.setattr(evaluate_module, "extract_answer", lambda text, task_type: text.strip())

    evaluate_module.evaluate_loaded_system(
        system=DummySystem(),
        config={"training": {"task": "gsm8k", "max_seq_len": 32}},
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

    progress_log = tmp_path / "eval_progress.log"
    text = progress_log.read_text(encoding="utf-8")
    assert "Running evaluation..." in text
    assert "Sample 1 [OK]" in text
    assert "EVALUATION RESULTS" in text


def test_run_system_batch_uses_nn_module_forward_signature():
    from src.cli.evaluate import run_system_batch

    class DummyBaseModel:
        def tokenize(self, texts, max_length=2048, add_special_tokens=True):
            batch = len(texts)
            return {
                "input_ids": torch.ones(batch, 3, dtype=torch.long),
                "attention_mask": torch.ones(batch, 3, dtype=torch.long),
            }

    class DummySystem(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = DummyBaseModel()
            self.seen = None

        def forward(
            self,
            task_token_ids,
            task_attention_mask=None,
            max_new_tokens=0,
            inference_mode="chat_with_prefix",
            use_terminal_prefix=True,
            communication_mode="latent_prefix",
            text_message_edge_threshold=0.5,
            text_message_max_new_tokens=512,
            do_sample=False,
            collect_agent_logs=False,
        ):
            self.seen = {
                "shape": tuple(task_token_ids.shape),
                "communication_mode": communication_mode,
                "text_message_edge_threshold": text_message_edge_threshold,
                "text_message_max_new_tokens": text_message_max_new_tokens,
            }
            batch = task_token_ids.shape[0]
            return {
                "generated_text": ["1"] * batch,
                "generation": {
                    "generated_text": ["1"] * batch,
                    "finish_reason": ["eos"] * batch,
                    "generated_token_count": [1] * batch,
                    "stopped_early": [True] * batch,
                },
                "agent_logs": [],
            }

    system = DummySystem()
    updates, _ = run_system_batch(
        system=system,
        config={"training": {"max_seq_len": 32}},
        batch={"questions": ["q1", "q2"], "answers": ["1", "1"]},
        task="gsm8k",
        device=torch.device("cpu"),
        generation_max_new_tokens=8,
        inference_mode="chat_with_prefix",
        use_terminal_prefix=False,
        communication_mode="text_messages",
        text_message_edge_threshold=0.7,
        text_message_max_new_tokens=256,
        do_sample=False,
        write_agent_logs=False,
    )

    assert len(updates) == 2
    assert system.seen == {
        "shape": (2, 3),
        "communication_mode": "text_messages",
        "text_message_edge_threshold": 0.7,
        "text_message_max_new_tokens": 256,
    }
