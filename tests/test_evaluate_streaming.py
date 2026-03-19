import json
from pathlib import Path


def test_streaming_eval_writes_jsonl_and_partial_snapshot(tmp_path: Path):
    from src.cli.evaluate import append_jsonl_record, write_partial_eval_snapshot

    jsonl_path = tmp_path / "eval_samples.jsonl"
    partial_path = tmp_path / "eval_results.partial.json"

    append_jsonl_record(
        jsonl_path,
        {
            "question": "q1",
            "gold": "1",
            "prediction": "1",
            "correct": True,
        },
    )
    append_jsonl_record(
        jsonl_path,
        {
            "question": "q2",
            "gold": "2",
            "prediction": "3",
            "correct": False,
        },
    )

    write_partial_eval_snapshot(
        partial_path=partial_path,
        method="ours_trained_multi_agent",
        task="gsm8k",
        correct=1,
        total=2,
        time_seconds=3.5,
        parameters={"max_samples": 2, "generation_max_new_tokens": 2048},
        world_size=2,
        jsonl_path=jsonl_path,
    )

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "q1"
    assert json.loads(lines[1])["question"] == "q2"

    payload = json.loads(partial_path.read_text())
    assert payload["method"] == "ours_trained_multi_agent"
    assert payload["metrics"]["correct"] == 1
    assert payload["metrics"]["total"] == 2
    assert payload["metrics"]["accuracy"] == 50.0
    assert payload["samples_jsonl"].endswith("eval_samples.jsonl")
