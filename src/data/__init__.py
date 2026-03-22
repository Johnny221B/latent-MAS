"""Dataset helpers exposed under src.data."""

from .base import MultiAgentDataset, build_labels
from .factory import create_dataset, get_task_configs

__all__ = [
    "MultiAgentDataset",
    "build_labels",
    "create_dataset",
    "get_task_configs",
]
