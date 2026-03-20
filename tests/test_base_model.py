"""Tests for BaseModelWrapper path resolution."""

import sys
from pathlib import Path
import torch.nn as nn
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_model import BaseModelWrapper


def test_resolve_model_path_uses_hf_default_cache_when_cache_dir_is_none():
    load_path, load_kwargs = BaseModelWrapper._resolve_model_path(
        "Qwen/Qwen3-0.6B",
        None,
    )
    assert load_path == "Qwen/Qwen3-0.6B"
    assert load_kwargs == {}


def test_set_trainable_reenables_requires_grad():
    wrapper = BaseModelWrapper.__new__(BaseModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = nn.Linear(4, 2)

    wrapper._freeze()
    assert all(not param.requires_grad for param in wrapper.model.parameters())

    wrapper.set_trainable(True)
    assert all(param.requires_grad for param in wrapper.model.parameters())


def test_resolve_dtype_maps_bfloat16_and_falls_back_to_float32():
    assert BaseModelWrapper._resolve_dtype("bfloat16") == torch.bfloat16
    assert BaseModelWrapper._resolve_dtype(None) == torch.float32


def test_get_input_embeddings_supports_ddp_wrapped_model():
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embedding

    class FakeDDP:
        def __init__(self, module):
            self.module = module

        def parameters(self):
            return self.module.parameters()

    wrapper = BaseModelWrapper.__new__(BaseModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = FakeDDP(FakeModel())

    embeds = wrapper.get_input_embeddings(torch.tensor([[1, 2]], dtype=torch.long))

    assert embeds.shape == (1, 2, 4)
