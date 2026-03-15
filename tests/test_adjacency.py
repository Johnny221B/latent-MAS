"""Tests for LearnableAdjacency."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.adjacency import LearnableAdjacency


def test_adjacency_init():
    """Adjacency should approximate the prior after initialization."""
    prior = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=torch.float32)

    adj = LearnableAdjacency(prior=prior, init_scale=5.0)
    A = adj.get_adjacency()

    # Prior edge (0->1) should be close to 1
    assert A[0, 1].item() > 0.99, f"Prior edge 0->1 should be ~1, got {A[0,1].item():.4f}"
    # Prior edge (1->2) should be close to 1
    assert A[1, 2].item() > 0.99, f"Prior edge 1->2 should be ~1, got {A[1,2].item():.4f}"
    # Non-prior edge (0->2) should be close to 0
    assert A[0, 2].item() < 0.01, f"Non-prior edge 0->2 should be ~0, got {A[0,2].item():.4f}"
    # Diagonal should be 0
    for i in range(3):
        assert A[i, i].item() < 1e-6, f"Diagonal [{i},{i}] should be 0"
    # Lower triangle should be 0
    assert A[1, 0].item() < 1e-6, "Lower triangle should be 0"
    assert A[2, 0].item() < 1e-6, "Lower triangle should be 0"

    print("✓ test_adjacency_init passed")


def test_adjacency_gradient():
    """Adjacency logits should receive gradients."""
    prior = torch.tensor([[0, 1], [0, 0]], dtype=torch.float32)
    adj = LearnableAdjacency(prior=prior)

    A = adj.get_adjacency()
    loss = A.sum()
    loss.backward()

    # logits[0,1] should have gradient (it's an upper-triangle prior edge)
    assert adj.logits.grad is not None, "Logits should have gradients"
    print("✓ test_adjacency_gradient passed")


def test_hard_adjacency():
    """Hard adjacency should binarize correctly."""
    prior = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=torch.float32)

    adj = LearnableAdjacency(prior=prior, init_scale=5.0)
    A_hard = adj.get_hard_adjacency(threshold=0.5)

    assert A_hard[0, 1] == 1.0
    assert A_hard[1, 2] == 1.0
    assert A_hard[0, 2] == 0.0
    print("✓ test_hard_adjacency passed")


if __name__ == "__main__":
    test_adjacency_init()
    test_adjacency_gradient()
    test_hard_adjacency()
    print("\nAll adjacency tests passed!")
