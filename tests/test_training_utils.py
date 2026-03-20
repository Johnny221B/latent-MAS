"""Tests for training batch validation helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.training import should_save_checkpoint, validate_min_samples_for_batches


def test_validate_min_samples_for_batches_rejects_zero_batch_ddp():
    with pytest.raises(ValueError, match="No training batches would be produced"):
        validate_min_samples_for_batches(
            dataset_size=8,
            per_gpu_batch_size=8,
            world_size=2,
            drop_last=True,
        )


def test_validate_min_samples_for_batches_allows_one_full_step_ddp():
    validate_min_samples_for_batches(
        dataset_size=16,
        per_gpu_batch_size=8,
        world_size=2,
        drop_last=True,
    )


def test_should_save_checkpoint_disables_non_positive_intervals():
    assert not should_save_checkpoint(global_step=32, save_interval=0)
    assert not should_save_checkpoint(global_step=32, save_interval=-1)


def test_should_save_checkpoint_matches_positive_interval_steps():
    assert should_save_checkpoint(global_step=32, save_interval=32)
    assert not should_save_checkpoint(global_step=31, save_interval=32)
