"""Tests for BaseModelWrapper path resolution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_model import BaseModelWrapper


def test_resolve_model_path_uses_hf_default_cache_when_cache_dir_is_none():
    load_path, load_kwargs = BaseModelWrapper._resolve_model_path(
        "Qwen/Qwen3-0.6B",
        None,
    )
    assert load_path == "Qwen/Qwen3-0.6B"
    assert load_kwargs == {}
