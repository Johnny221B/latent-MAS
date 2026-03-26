"""Dataset registry and factory helpers."""

from .am_deepseek_r1_distilled import build_task_configs as build_am_deepseek_r1_distilled_task_configs
from .arc import build_task_configs as build_arc_task_configs
from .base import MultiAgentDataset
from .competition_math import build_task_configs as build_competition_math_task_configs
from .gsm8k import build_task_configs as build_gsm8k_task_configs
from .humaneval import build_task_configs as build_humaneval_task_configs


def get_task_configs() -> dict:
    task_configs = {}
    task_configs.update(build_am_deepseek_r1_distilled_task_configs())
    task_configs.update(build_gsm8k_task_configs())
    task_configs.update(build_arc_task_configs())
    task_configs.update(build_competition_math_task_configs())
    task_configs.update(build_humaneval_task_configs())
    return task_configs


def create_dataset(task: str, split: str = "train", max_samples: int | None = None) -> MultiAgentDataset:
    return MultiAgentDataset(task=task, split=split, max_samples=max_samples)
