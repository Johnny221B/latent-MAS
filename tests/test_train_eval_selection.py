import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_build_live_eval_plan_skips_zero_sample_splits():
    from src.cli.train import build_live_eval_plan

    plan = build_live_eval_plan(
        {
            "train_probe_samples": 0,
            "test_probe_samples": 64,
        }
    )

    assert plan == [("test", 64)]


def test_build_live_eval_plan_keeps_nonzero_train_split():
    from src.cli.train import build_live_eval_plan

    plan = build_live_eval_plan(
        {
            "train_probe_samples": 32,
            "test_probe_samples": 64,
        }
    )

    assert plan == [("train", 32), ("test", 64)]
