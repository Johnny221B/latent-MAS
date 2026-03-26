"""Tests for config normalization and compatibility shims."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config


def _write_minimal_graph(tmp_path: Path) -> Path:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "agents": ["planner", "solver", "critic"],
                "adjacency_prior": [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ],
                "execution_order": [0, 1, 2],
                "terminal_agent_index": 2,
            }
        ),
        encoding="utf-8",
    )
    return graph_path


def test_load_config_normalizes_train_strategy_from_deprecated_alias(tmp_path: Path):
    graph_path = _write_minimal_graph(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "dummy"},
                "graph": {"config": str(graph_path)},
                "training": {
                    "task": "gsm8k",
                    "batch_size": 32,
                    "train_base_model": True,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded["training"]["train_strategy"] == "full_finetune"


def test_load_config_defaults_train_strategy_to_communication_only(tmp_path: Path):
    graph_path = _write_minimal_graph(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "dummy"},
                "graph": {"config": str(graph_path)},
                "training": {
                    "task": "gsm8k",
                    "batch_size": 32,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded["training"]["train_strategy"] == "communication_only"


def test_load_config_defaults_training_shuffle_to_true(tmp_path: Path):
    graph_path = _write_minimal_graph(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "dummy"},
                "graph": {"config": str(graph_path)},
                "training": {
                    "task": "gsm8k",
                    "batch_size": 32,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded["training"]["shuffle"] is True


def test_load_config_defaults_training_seed_and_probe_degeneracy_ratio(tmp_path: Path):
    graph_path = _write_minimal_graph(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "dummy"},
                "graph": {"config": str(graph_path)},
                "training": {
                    "task": "gsm8k",
                    "batch_size": 32,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded["training"]["seed"] == 42
    assert loaded["training_probe"]["degenerate_max_new_tokens_ratio"] == 0.5


def test_probe64_configs_disable_final_checkpoint_and_enable_live_eval():
    repo_root = Path(__file__).resolve().parent.parent
    config_paths = [
        repo_root / "configs/experiments/gsm8k_5agent_probe64_comm_only.yaml",
        repo_root / "configs/experiments/gsm8k_5agent_probe64_full_finetune.yaml",
    ]
    existing_paths = [config_path for config_path in config_paths if config_path.exists()]

    if not existing_paths:
        return

    for config_path in existing_paths:
        loaded = load_config(config_path)
        assert loaded["training"]["batch_size"] == 32
        assert loaded["training"]["save_final_checkpoint"] is False
        assert loaded["evaluation"]["run_after_train"] is True


def test_humaneval_experiment_config_declares_pass_at_k_settings():
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "configs/experiments/humaneval_5agent.yaml"

    loaded = load_config(config_path)

    assert loaded["training"]["task"] == "humaneval"
    assert loaded["evaluation"]["metric"] == "pass_at_k"
    assert loaded["evaluation"]["split_scheme"] == "debug_60_40"
    assert loaded["evaluation"]["num_samples_per_task"] == 20
    assert loaded["evaluation"]["pass_at_k"] == [1, 10]
    assert loaded["evaluation"]["do_sample"] is True


def test_arc_experiment_configs_enable_post_train_test_eval():
    repo_root = Path(__file__).resolve().parent.parent
    config_paths = [
        repo_root / "configs/experiments/arc_easy_5agent.yaml",
        repo_root / "configs/experiments/arc_challenge_5agent.yaml",
    ]

    for config_path in config_paths:
        loaded = load_config(config_path)
        assert loaded["training"]["task"] in {"arc_easy", "arc_challenge"}
        assert loaded["evaluation"]["run_after_train"] is True
        assert loaded["evaluation"]["splits_after_train"] == ["test"]


def test_competition_math_experiment_configs_use_probe_monitoring():
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "configs/experiments/competition_math_5agent.yaml"
    debug_config_path = repo_root / "configs/experiments/competition_math_5agent_debug.yaml"

    loaded = load_config(config_path)
    assert loaded["training"]["task"] == "competition_math"
    assert loaded["training"]["train_strategy"] == "communication_only"
    assert loaded["evaluation"]["run_after_train"] is False
    assert loaded["training_probe"]["enabled"] is True
    assert loaded["training_probe"]["samples"] >= 0

    debug_loaded = load_config(debug_config_path)
    assert debug_loaded["training"]["task"] == "competition_math"
    assert debug_loaded["training_probe"]["enabled"] is True
    assert debug_loaded["training_probe"]["samples"] == 100
    assert debug_loaded["report"]["use_wandb"] is False


def test_am_deepseek_r1_distilled_experiment_config_matches_competition_math_shape():
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "configs/experiments/am_deepseek_r1_distilled_5agent.yaml"

    loaded = load_config(config_path)

    assert loaded["training"]["task"] == "am_deepseek_r1_distilled"
    assert loaded["training"]["train_strategy"] == "communication_only"
    assert loaded["training"]["input_mode"] == "chat_with_prefix"
    assert loaded["training"]["batch_size"] == 4
    assert loaded["training"]["max_seq_len"] == 8192
    assert loaded["training_probe"]["enabled"] is True
    assert loaded["training_probe"]["samples"] == 0
    assert loaded["evaluation"]["run_after_train"] is False
    assert loaded["model"]["name"] == "Qwen/Qwen3-4B"
    assert loaded["model"]["dtype"] == "bfloat16"
    assert loaded["output"]["dir"] == "outputs/am_deepseek_r1_distilled_qwen3-4b"
    assert loaded["report"]["use_wandb"] is True


def test_qwen3_4b_smoke_configs_are_small_and_disable_wandb():
    repo_root = Path(__file__).resolve().parent.parent
    config_paths = [
        repo_root / "configs/experiments/arc_easy_5agent_qwen3_4b_smoke.yaml",
        repo_root / "configs/experiments/humaneval_5agent_qwen3_4b_smoke.yaml",
    ]

    for config_path in config_paths:
        loaded = load_config(config_path)
        assert loaded["model"]["name"] == "Qwen/Qwen3-4B"
        assert loaded["model"]["dtype"] == "bfloat16"
        assert loaded["training"]["batch_size"] == 1
        assert loaded["report"]["use_wandb"] is False
