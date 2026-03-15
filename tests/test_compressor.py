"""Tests for the LatentCompressor."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.compressor import LatentCompressor


def test_compressor_output_shape():
    """Output should always be [B, Lp, D] regardless of input length."""
    D = 64
    Lp = 8
    comp = LatentCompressor(hidden_dim=D, num_queries=Lp, num_heads=4)

    # Variable input lengths
    for seq_len in [10, 50, 200]:
        x = torch.randn(2, seq_len, D)
        out = comp(x)
        assert out.shape == (2, Lp, D), f"Expected (2, {Lp}, {D}), got {out.shape}"
    print("✓ test_compressor_output_shape passed")


def test_compressor_gradient_flow():
    """Gradients should flow through the compressor."""
    D = 64
    comp = LatentCompressor(hidden_dim=D, num_queries=8, num_heads=4)

    x = torch.randn(2, 20, D, requires_grad=True)
    out = comp(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients should flow to input"
    assert comp.queries.grad is not None, "Gradients should flow to queries"
    print("✓ test_compressor_gradient_flow passed")


def test_compressor_with_mask():
    """Compressor should handle attention masks correctly."""
    D = 64
    comp = LatentCompressor(hidden_dim=D, num_queries=8, num_heads=4)

    x = torch.randn(2, 20, D)
    mask = torch.ones(2, 20)
    mask[0, 15:] = 0  # mask out last 5 tokens for first sample
    mask[1, 10:] = 0  # mask out last 10 tokens for second sample

    out = comp(x, mask=mask)
    assert out.shape == (2, 8, D)
    print("✓ test_compressor_with_mask passed")


if __name__ == "__main__":
    test_compressor_output_shape()
    test_compressor_gradient_flow()
    test_compressor_with_mask()
    print("\nAll compressor tests passed!")
