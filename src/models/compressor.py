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
