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


def test_resolve_resume_checkpoint_returns_none_without_config():
    from src.cli.train import resolve_resume_checkpoint

    assert resolve_resume_checkpoint({"resume_from_checkpoint": None}) is None
    assert resolve_resume_checkpoint({}) is None


def test_resolve_resume_checkpoint_returns_path_when_configured(tmp_path: Path):
    from src.cli.train import resolve_resume_checkpoint

    checkpoint_path = tmp_path / "run" / "checkpoint_step3000.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")

    resolved = resolve_resume_checkpoint(
        {"resume_from_checkpoint": str(checkpoint_path)}
    )

    assert resolved == checkpoint_path.resolve()


def test_resolve_training_output_dir_uses_resume_checkpoint_parent(tmp_path: Path):
    from src.cli.train import resolve_training_output_dir

    checkpoint_path = tmp_path / "run" / "checkpoint_step3000.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    config = {
        "training": {"resume_from_checkpoint": str(checkpoint_path)},
        "output": {"dir": "outputs/ignored"},
    }

    assert resolve_training_output_dir(config) == checkpoint_path.parent.resolve()


def test_compute_resume_schedule_uses_checkpoint_progress_when_present():
    from src.cli.train import compute_resume_schedule

    schedule = compute_resume_schedule(
        checkpoint={"step": 3000, "epoch": 0, "next_batch_idx": 3000},
        batches_per_epoch=5000,
        grad_accum_steps=1,
    )

    assert schedule["global_step"] == 3000
    assert schedule["start_epoch"] == 0
    assert schedule["skip_batches_in_epoch"] == 3000


def test_compute_resume_schedule_falls_back_to_global_step_and_grad_accum():
    from src.cli.train import compute_resume_schedule

    schedule = compute_resume_schedule(
        checkpoint={"step": 7},
        batches_per_epoch=10,
        grad_accum_steps=2,
    )

    assert schedule["global_step"] == 7
    assert schedule["start_epoch"] == 1
    assert schedule["skip_batches_in_epoch"] == 4


def test_load_training_checkpoint_restores_trainable_states(tmp_path: Path):
    from src.cli.train import load_training_checkpoint

    source_base = torch.nn.Linear(2, 2)
    source_compressor = torch.nn.Linear(2, 2)
    source_adjacency = torch.nn.Linear(2, 2)
    source_optimizer = torch.optim.AdamW(source_compressor.parameters(), lr=0.1)
    source_compressor.weight.grad = torch.ones_like(source_compressor.weight)
    source_compressor.bias.grad = torch.ones_like(source_compressor.bias)
    source_optimizer.step()

    checkpoint_path = tmp_path / "checkpoint_step3000.pt"
    torch.save(
        {
            "step": 3000,
            "epoch": 0,
            "next_batch_idx": 3000,
            "base_model_state": source_base.state_dict(),
            "compressor_state": source_compressor.state_dict(),
            "adjacency_state": source_adjacency.state_dict(),
            "optimizer_state": source_optimizer.state_dict(),
        },
        checkpoint_path,
    )

    target_base = torch.nn.Linear(2, 2)
    target_compressor = torch.nn.Linear(2, 2)
    target_adjacency = torch.nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(target_compressor.parameters(), lr=0.1)

    loaded = load_training_checkpoint(
        checkpoint_path=checkpoint_path,
        compressor=target_compressor,
        adjacency=target_adjacency,
        optimizer=target_optimizer,
        base_model_module=target_base,
        trainable_base_model=True,
    )

    assert loaded["global_step"] == 3000
    assert loaded["start_epoch"] == 0
    assert loaded["skip_batches_in_epoch"] == 3000
    assert target_optimizer.state_dict()["state"]

    for source_param, target_param in zip(source_base.parameters(), target_base.parameters(), strict=True):
        assert torch.equal(source_param, target_param)
    for source_param, target_param in zip(source_compressor.parameters(), target_compressor.parameters(), strict=True):
        assert torch.equal(source_param, target_param)
    for source_param, target_param in zip(source_adjacency.parameters(), target_adjacency.parameters(), strict=True):
        assert torch.equal(source_param, target_param)


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
