import json
from pathlib import Path


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
