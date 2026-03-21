"""Tests for LearnableAdjacency."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.graph.adjacency import (
    LearnableAdjacency,
    build_topological_edge_mask,
    validate_graph_topology,
)
from src.losses.graph_loss import GraphLoss


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


def test_build_topological_edge_mask_supports_custom_execution_order():
    mask = build_topological_edge_mask(
        execution_order=[0, 2, 1],
        terminal_agent_index=1,
    )

    expected = torch.tensor([
        [False, True, True],
        [False, False, False],
        [False, True, False],
    ])
    assert torch.equal(mask.cpu(), expected)


def test_adjacency_can_allow_edges_that_are_backward_by_raw_index():
    prior = torch.tensor([
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)
    allowed_mask = build_topological_edge_mask(
        execution_order=[0, 2, 1],
        terminal_agent_index=1,
    )

    adj = LearnableAdjacency(prior=prior, allowed_edges_mask=allowed_mask, init_scale=5.0)
    A_hard = adj.get_hard_adjacency(threshold=0.5)

    assert A_hard[0, 2] == 1.0
    assert A_hard[2, 1] == 1.0
    assert A_hard[1, 2] == 0.0


def test_graph_loss_respects_custom_allowed_edges_mask():
    adjacency = torch.tensor([
        [0.0, 0.7, 0.8],
        [0.0, 0.0, 0.9],
        [0.0, 0.6, 0.0],
    ])
    prior = torch.tensor([
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)
    allowed_mask = build_topological_edge_mask(
        execution_order=[0, 2, 1],
        terminal_agent_index=1,
    )

    out = GraphLoss(lambda_add=1.0, lambda_drop=1.0, lambda_sparse=1.0)(
        adjacency=adjacency,
        prior=prior,
        valid_mask=allowed_mask,
    )

    assert torch.isclose(out["loss_add"], torch.tensor(0.7))
    assert torch.isclose(out["loss_drop"], torch.tensor(0.6))
    assert torch.isclose(out["loss_sparse"], torch.tensor(2.1))


def test_adjacency_init_scale_is_configurable():
    prior = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=torch.float32)

    strong = LearnableAdjacency(prior=prior, init_scale=5.0)
    weak = LearnableAdjacency(prior=prior, init_scale=1.5)

    strong_A = strong.get_adjacency()
    weak_A = weak.get_adjacency()

    assert strong_A[0, 1] > weak_A[0, 1]
    assert strong_A[0, 2] < weak_A[0, 2]


def test_prior_residual_parameterization_starts_from_prior_logits():
    prior = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=torch.float32)

    adj = LearnableAdjacency(
        prior=prior,
        init_scale=1.5,
        parameterization="prior_residual",
    )
    A = adj.get_adjacency()

    assert A[0, 1] > 0.7
    assert A[0, 2] < 0.3


def test_validate_graph_topology_rejects_prior_edges_that_violate_execution_order():
    prior = torch.tensor([
        [0, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
    ], dtype=torch.float32)

    with pytest.raises(ValueError, match="execution_order"):
        validate_graph_topology(
            prior=prior,
            execution_order=[0, 2, 1],
            terminal_agent_index=1,
        )


if __name__ == "__main__":
    test_adjacency_init()
    test_adjacency_gradient()
    test_hard_adjacency()
    print("\nAll adjacency tests passed!")
