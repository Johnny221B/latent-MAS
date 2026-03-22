"""Tests for train CLI helpers."""

import sys
from collections import Counter
from pathlib import Path

import pytest

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
