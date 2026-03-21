"""Tests for automatic communication-only graph search planning."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.run_comm_graph_search import (
    EvalWatchdogState,
    analyze_eval_progress_log,
    build_stage1_specs,
    build_stage2_specs,
    build_train_command,
    discover_hf_home,
    discover_python_bin,
    find_iteration_output_dir,
    maybe_abort_for_eval_gibberish,
    prepare_iteration_config,
    run_search,
)


def test_stage1_specs_cover_five_graph_first_iterations():
    specs = build_stage1_specs()

    assert len(specs) >= 5
    assert specs[0].slug == "baseline"
    assert any(spec.slug == "lambda-rebalance" for spec in specs)
    assert any(spec.slug == "low-init-scale" for spec in specs)
    assert any(spec.slug == "prior-residual" for spec in specs)


def test_prepare_iteration_config_enforces_comm_only_probe64_invariants(tmp_path: Path):
    base_config = {
        "model": {"name": "dummy"},
        "graph": {"config": "configs/graphs/default_5agent.json", "roles_dir": "configs/roles"},
        "compressor": {"num_queries": 16},
        "training": {
            "task": "gsm8k",
            "train_strategy": "full_finetune",
            "batch_size": 8,
            "epochs": 64,
        },
        "evaluation": {
            "run_after_train": False,
            "write_agent_logs": False,
        },
        "output": {"dir": "outputs/old"},
    }
    spec = build_stage1_specs()[0]

    updated = prepare_iteration_config(
        base_config=base_config,
        spec=spec,
        campaign_root=tmp_path / "campaign",
    )

    assert updated["training"]["train_strategy"] == "communication_only"
    assert updated["training"]["batch_size"] == 32
    assert updated["evaluation"]["run_after_train"] is True
    assert updated["evaluation"]["train_probe_samples"] == 0
    assert updated["evaluation"]["test_probe_samples"] == 64
    assert updated["evaluation"]["write_agent_logs"] is True
    assert str(updated["output"]["dir"]).startswith(str(tmp_path / "campaign" / "iter01_"))


def test_stage2_specs_expand_without_enabling_backbone_training():
    specs = build_stage2_specs()

    assert len(specs) >= 3
    assert all("train_strategy" not in spec.overrides.get("training", {}) for spec in specs)


def test_build_train_command_uses_torchrun_and_max_samples():
    command = build_train_command(
        config_path=Path("configs/run.yaml"),
        max_samples=64,
        nproc_per_node=2,
        python_bin=".venv/bin/python",
    )

    assert command[:5] == ["uv", "run", "--python", ".venv/bin/python", "torchrun"]
    assert "--config" in command
    assert "--max_samples" in command
    assert "64" in command


def test_find_iteration_output_dir_prefers_new_timestamped_run(tmp_path: Path):
    base_output_dir = tmp_path / "iter01_baseline"
    older = tmp_path / "iter01_baseline_20260320_010101"
    newer = tmp_path / "iter01_baseline_20260320_020202"
    older.mkdir()
    newer.mkdir()

    resolved = find_iteration_output_dir(base_output_dir, existing_paths={older.resolve()})

    assert resolved == newer.resolve()


def test_discover_python_bin_falls_back_to_explicit_path():
    assert discover_python_bin("/tmp/custom-python") == "/tmp/custom-python"


def test_discover_hf_home_falls_back_to_explicit_path():
    assert discover_hf_home("/tmp/custom-hf") == "/tmp/custom-hf"


def test_analyze_eval_progress_log_flags_gibberish_patterns():
    reason = analyze_eval_progress_log(
        """
Running evaluation...
  Sample 1 [OK]
    Pred: <empty>
    Finish: max_new_tokens | Tokens: 12000
    Gen: .
"""
    )

    assert reason is not None
    assert "max_new_tokens" in reason


def test_maybe_abort_for_eval_gibberish_waits_until_threshold(tmp_path: Path):
    progress_log = tmp_path / "eval_progress.log"
    progress_log.write_text(
        """
Running evaluation...
  Sample 1 [OK]
    Pred: <empty>
    Finish: max_new_tokens | Tokens: 12000
    Gen: .
""".strip()
        + "\n",
        encoding="utf-8",
    )
    state = EvalWatchdogState()

    early = maybe_abort_for_eval_gibberish(
        progress_log=progress_log,
        state=state,
        now=1000.0,
        threshold_seconds=1800.0,
    )
    late = maybe_abort_for_eval_gibberish(
        progress_log=progress_log,
        state=state,
        now=2801.0,
        threshold_seconds=1800.0,
    )

    assert early is None
    assert late is not None
    assert "gibberish" in late.lower()


def test_run_search_records_iteration_failures_and_keeps_going(tmp_path: Path, monkeypatch):
    import subprocess
    from src.cli import run_comm_graph_search as runner

    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(
        """
model:
  name: dummy
graph:
  config: configs/graphs/default_5agent.json
  roles_dir: configs/roles
compressor:
  num_queries: 16
training:
  task: gsm8k
  epochs: 1
evaluation: {}
output:
  dir: outputs/base
""".strip()
        + "\n",
        encoding="utf-8",
    )

    specs = runner.build_stage1_specs()[:2]
    monkeypatch.setattr(runner, "build_stage1_specs", lambda: specs)
    monkeypatch.setattr(runner, "build_stage2_specs", lambda: [])

    calls = {"count": 0}

    def fake_run_training_iteration(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise subprocess.CalledProcessError(returncode=1, cmd=["torchrun"])
        output_dir = tmp_path / f"iter{calls['count']:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "output_dir": str(output_dir),
            "train_summary": {"metrics": {"accuracy": 55.0}},
            "test_summary": {"metrics": {"accuracy": 81.0}},
        }

    monkeypatch.setattr(runner, "run_training_iteration", fake_run_training_iteration)

    result = run_search(
        base_config_path=str(base_config_path),
        campaign_root=str(tmp_path / "campaign"),
        target_accuracy=80.0,
        min_iterations=2,
        max_samples=64,
        launcher="torchrun",
        nproc_per_node=2,
        python_bin="/tmp/python",
        hf_home=None,
        offline=True,
        cuda_visible_devices=None,
    )

    assert len(result["iterations"]) == 2
    assert result["iterations"][0]["status"] == "failed"
    assert result["iterations"][0]["test_accuracy"] == 0.0
    assert result["iterations"][1]["status"] == "completed"
    leaderboard = json.loads((tmp_path / "campaign" / "leaderboard.json").read_text())
    assert leaderboard[0]["status"] == "failed"
    assert leaderboard[1]["test_accuracy"] == 81.0
