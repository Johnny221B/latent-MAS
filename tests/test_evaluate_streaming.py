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
