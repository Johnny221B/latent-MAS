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
    ):
        """
        Args:
            hidden_dim: dimension of hidden states (must match base model)
            num_queries: Lp, number of output prefix tokens
            num_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Learnable query tokens: these become the output prefix
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)

        # Single-layer cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm and FFN (standard transformer block components)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.target_norm = target_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress variable-length hidden states into fixed-length prefix.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
                The latent trajectory S_i from an upstream agent.
            mask: [batch_size, seq_len] optional attention mask
                1 = attend, 0 = ignore (e.g., for padding)

        Returns:
            prefix: [batch_size, num_queries, hidden_dim]
                Fixed-length latent prefix to be sent downstream.
        """
        B = hidden_states.shape[0]
        compute_dtype = self.queries.dtype
        if hidden_states.dtype != compute_dtype:
            hidden_states = hidden_states.to(dtype=compute_dtype)

        # Expand queries for the batch
        queries = self.queries.expand(B, -1, -1)  # [B, Lp, D]

        # Convert mask to key_padding_mask format if provided
        # nn.MultiheadAttention expects: True = ignore, False = attend
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # invert: 1->False(attend), 0->True(ignore)

        # Cross-attention: queries attend to hidden states
        attn_out, _ = self.cross_attn(
            query=queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )

        # Residual + LayerNorm
        x = self.norm1(queries + attn_out)

        # FFN + Residual + LayerNorm
        x = self.norm2(x + self.ffn(x))

        # Scale output to match input embedding norm
        # Without this, prefix norm (~100) >> embedding norm (~1-2),
        # causing the frozen LLM to behave erratically.
        if self.target_norm is not None:
            x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x = x * (self.target_norm / x_norm)

        return x  # [B, Lp, D]

        return x  # [B, Lp, D]

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
