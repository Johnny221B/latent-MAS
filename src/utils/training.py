"""Training-time validation helpers."""

import math

import torch


def validate_min_samples_for_batches(
    dataset_size: int,
    per_gpu_batch_size: int,
    world_size: int = 1,
    drop_last: bool = True,
) -> None:
    """Fail early when the current batch config would yield zero steps."""
    if not drop_last:
        return

    required_samples = per_gpu_batch_size * world_size
    if dataset_size < required_samples:
        raise ValueError(
            "No training batches would be produced: "
            f"dataset_size={dataset_size}, per_gpu_batch_size={per_gpu_batch_size}, "
            f"world_size={world_size}, drop_last={drop_last}. "
            f"Need at least {required_samples} samples."
        )


def compute_grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().norm(2).item()
        total += grad_norm * grad_norm
    return math.sqrt(total)


def should_save_checkpoint(global_step: int, save_interval: int) -> bool:
    """Return whether the current step should emit an intermediate checkpoint."""
    if save_interval <= 0:
        return False
    return global_step > 0 and global_step % save_interval == 0
