"""Tests for train CLI helpers."""

import random
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_resolve_post_train_eval_plan_uses_configured_splits():
    from src.cli.train import resolve_post_train_eval_plan

    evaluation_cfg = {
        "splits_after_train": ["test", "train"],
        "test_probe_samples": 64,
        "train_probe_samples": 16,
    }

    assert resolve_post_train_eval_plan(evaluation_cfg) == [
        ("test", 64),
        ("train", 16),
    ]


def test_resolve_post_train_eval_plan_defaults_to_train_then_test():
    from src.cli.train import resolve_post_train_eval_plan

    assert resolve_post_train_eval_plan({}) == [
        ("train", None),
        ("test", None),
    ]


def test_split_training_and_probe_subsets_holds_out_fixed_probe_size():
    from src.cli.train import split_training_and_probe_subsets

    dataset = list(range(10))
    train_subset, probe_subset, probe_indices = split_training_and_probe_subsets(
        dataset,
        probe_samples=3,
        seed=7,
    )

    assert len(train_subset) == 7
    assert len(probe_subset) == 3
    combined = list(train_subset.indices) + list(probe_subset.indices)
    assert Counter(combined) == Counter(range(10))
    assert probe_indices == sorted(probe_subset.indices)


def test_split_training_and_probe_subsets_rejects_full_holdout():
    from src.cli.train import split_training_and_probe_subsets

    with pytest.raises(ValueError, match="Probe split requires more samples than available"):
        split_training_and_probe_subsets(
            list(range(5)),
            probe_samples=5,
            seed=42,
        )


def test_should_run_training_probe_uses_step_interval():
    from src.cli.train import should_run_training_probe

    cfg = {"enabled": True, "every_n_steps": 10}
    assert not should_run_training_probe(9, cfg)
    assert should_run_training_probe(10, cfg)
    assert not should_run_training_probe(10, {"enabled": False, "every_n_steps": 10})


def test_set_global_seed_controls_python_and_torch_rng():
    from src.cli.train import set_global_seed

    set_global_seed(123)
    first = (random.random(), torch.rand(3).tolist())

    set_global_seed(123)
    second = (random.random(), torch.rand(3).tolist())

    assert first == second


def test_build_run_provenance_captures_launch_metadata(monkeypatch, tmp_path: Path):
    from src.cli import train as train_module

    monkeypatch.setattr(
        train_module,
        "collect_git_provenance",
        lambda repo_root=None: {
            "commit": "abc123",
            "branch": "toby",
            "status_short": [" M src/cli/train.py"],
            "diff_stat": [" src/cli/train.py | 10 +++++++++-"],
        },
    )

    provenance = train_module.build_run_provenance(
        config_path="configs/experiments/gsm8k_5agent.yaml",
        output_dir=tmp_path,
        training_seed=123,
        world_size=2,
        rank=0,
        is_ddp=True,
        argv=["python", "src/cli/train.py", "--config", "cfg.yaml"],
        env={
            "CUDA_VISIBLE_DEVICES": "0,1",
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "2",
        },
    )

    assert provenance["training"]["seed"] == 123
    assert provenance["launch"]["argv"] == ["python", "src/cli/train.py", "--config", "cfg.yaml"]
    assert provenance["launch"]["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert provenance["launch"]["world_size"] == 2
    assert provenance["git"]["commit"] == "abc123"
    assert provenance["output_dir"] == str(tmp_path)


def test_summarize_probe_generation_flags_max_token_degeneracy():
    from src.cli.train import summarize_probe_generation_health

    summary = summarize_probe_generation_health(
        [
            {"generation": {"generated_token_count": 64, "finish_reason": "max_new_tokens"}},
            {"generation": {"generated_token_count": 64, "finish_reason": "max_new_tokens"}},
            {"generation": {"generated_token_count": 12, "finish_reason": "eos"}},
        ],
        max_new_tokens=64,
        degeneracy_ratio=0.5,
    )

    assert summary["max_new_tokens_count"] == 2
    assert summary["max_new_tokens_ratio"] == pytest.approx(2 / 3)
    assert summary["degenerate"] is True


def test_summarize_probe_generation_ignores_healthy_mix():
    from src.cli.train import summarize_probe_generation_health

    summary = summarize_probe_generation_health(
        [
            {"generation": {"generated_token_count": 10, "finish_reason": "eos"}},
            {"generation": {"generated_token_count": 9, "finish_reason": "stop"}},
            {"generation": {"generated_token_count": 64, "finish_reason": "max_new_tokens"}},
        ],
        max_new_tokens=64,
        degeneracy_ratio=0.8,
    )

    assert summary["max_new_tokens_count"] == 1
    assert summary["degenerate"] is False
