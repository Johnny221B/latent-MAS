# src/models/compressor.py
"""
LatentCompressor: compresses variable-length hidden states into fixed-length prefix.

Architecture (Q-Former style):
  - Lp learnable query tokens
  - Single-layer cross-attention: queries attend to input hidden states
  - Output: [batch_size, Lp, hidden_dim] — fixed-length prefix

This is the ONLY trainable component that touches information flow between agents.
All agents currently share a single compressor instance.

Extensibility:
  - To have per-edge compressors, create a CompressorBank indexed by (role_i, role_j).
  - To switch to Option 2 (KV cache) or Option 3 (memory bank), replace this module.
"""

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache


class HiddenProjection(nn.Module):
    """Projects hidden states from one dimension to another.

    Used in heterogeneous setups where small-model agents produce hidden states
    with a different dimension than the canonical communication dimension.
    """

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(source_dim, target_dim),
            nn.LayerNorm(target_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, S, D_source] -> [B, S, D_target]"""
        hidden_states = hidden_states.to(self.proj[0].weight.dtype)
        return self.proj(hidden_states)


class LatentCompressor(nn.Module):
    """Cross-attention compressor: variable-length input -> fixed-length prefix."""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        target_norm: float | None = None,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers

        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm1": nn.LayerNorm(hidden_dim),
                "norm2": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
            }))

        self.target_norm = target_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = hidden_states.shape[0]
        compute_dtype = self.queries.dtype
        if hidden_states.dtype != compute_dtype:
            hidden_states = hidden_states.to(dtype=compute_dtype)

        x = self.queries.expand(B, -1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)

        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](
                query=x,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=key_padding_mask,
            )
            x = layer["norm1"](x + attn_out)
            x = layer["norm2"](x + layer["ffn"](x))

        if self.target_norm is not None:
            x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x = x * (self.target_norm / x_norm)

        return x


class PrefixProjector(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_kv_heads: int,
        head_dim: int,
        cache_config=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cache_config = cache_config
        self.kv_dim = num_kv_heads * head_dim  # 对于 Qwen3-4B: 8 * 80 = 640
        
        # 使用 MLP 重参数化可以提升训练稳定性 (参考 Prefix-Tuning 论文)
        # 这里使用两层 MLP 映射到每一层的 K 和 V
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_layers * 2 * self.kv_dim)
        )

    def forward(self, prefix_embeds: torch.Tensor, target_dtype: torch.dtype | None = None) -> DynamicCache:
        """
        Args:
            prefix_embeds: [B, Lp, hidden_dim] (即 aggregator 输出的 z_j)
            target_dtype: if set, cast KV cache to this dtype (e.g. model's dtype)
        Returns:
            DynamicCache 兼容的 past_key_values
        """
        B, Lp, _ = prefix_embeds.shape

        # [B, Lp, num_layers * 2 * kv_dim]
        kv_out = self.mlp(prefix_embeds)

        # Cast to model dtype if needed (e.g. projector is fp32 but model is bf16)
        if target_dtype is not None and kv_out.dtype != target_dtype:
            kv_out = kv_out.to(dtype=target_dtype)

        # Reshape: [B, Lp, num_layers, 2, num_kv_heads, head_dim]
        kv_out = kv_out.view(B, Lp, self.num_layers, 2, self.num_kv_heads, self.head_dim)

        # 调换维度以便后续分离: -> [num_layers, 2, B, num_kv_heads, Lp, head_dim]
        kv_out = kv_out.permute(2, 3, 0, 4, 1, 5)

        try:
            cache = DynamicCache(config=self.cache_config) if self.cache_config is not None else DynamicCache()
        except TypeError:
            cache = DynamicCache()
        for layer_idx in range(self.num_layers):
            k = kv_out[layer_idx, 0]  # [B, num_kv_heads, Lp, head_dim]
            v = kv_out[layer_idx, 1]  # [B, num_kv_heads, Lp, head_dim]
            # 为了兼容性，使用 contiguous
            cache.update(k.contiguous(), v.contiguous(), layer_idx)

        return cache

# 上游 hidden trajectory [B, S, D]
#         ↓
# learnable queries [B, Lp, D]
#         ↓
# cross-attention: queries attend to hidden_states
#         ↓
# 得到 attn_out [B, Lp, D]
#         ↓
# residual + norm + FFN
#         ↓
# 输出 prefix [B, Lp, D]
