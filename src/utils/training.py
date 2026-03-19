"""Training-time validation helpers."""


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
