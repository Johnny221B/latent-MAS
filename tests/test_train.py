"""Tests for train CLI helpers."""

import sys
from pathlib import Path

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
