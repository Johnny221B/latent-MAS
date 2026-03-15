"""Tests for DAGExecutor and MessageAggregator."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.communication.aggregator import MessageAggregator


def test_aggregator_no_upstream():
    """First agent should get None prefix."""
    agg = MessageAggregator()
    result = agg.aggregate(agent_index=0, adjacency=torch.eye(3), all_prefixes=[])
    assert result is None, "Agent 0 should have no upstream prefix"
    print("✓ test_aggregator_no_upstream passed")


def test_aggregator_weighted_sum():
    """Aggregated prefix should be weighted sum of upstream prefixes."""
    agg = MessageAggregator()
    B, Lp, D = 2, 4, 8

    # Two upstream agents with known prefixes
    P0 = torch.ones(B, Lp, D)       # all ones
    P1 = torch.ones(B, Lp, D) * 2   # all twos

    adjacency = torch.tensor([
        [0, 0, 0.5],
        [0, 0, 0.3],
        [0, 0, 0],
    ])

    result = agg.aggregate(
        agent_index=2,
        adjacency=adjacency,
        all_prefixes=[P0, P1],
    )

    # Expected: 0.5 * P0 + 0.3 * P1 = 0.5 * 1 + 0.3 * 2 = 1.1
    expected = 0.5 * P0 + 0.3 * P1
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected[0,0,0]}, got {result[0,0,0]}"
    print("✓ test_aggregator_weighted_sum passed")


def test_aggregator_gradient_through_weights():
    """Gradients should flow through adjacency weights."""
    agg = MessageAggregator()
    B, Lp, D = 2, 4, 8

    P0 = torch.randn(B, Lp, D)

    # Make adjacency require grad (simulating LearnableAdjacency)
    adjacency = torch.tensor([
        [0, 0.8],
        [0, 0],
    ], requires_grad=True)

    result = agg.aggregate(agent_index=1, adjacency=adjacency, all_prefixes=[P0])
    loss = result.sum()
    loss.backward()

    assert adjacency.grad is not None, "Adjacency should receive gradients"
    assert adjacency.grad[0, 1].item() != 0, "Edge weight gradient should be non-zero"
    print("✓ test_aggregator_gradient_through_weights passed")


if __name__ == "__main__":
    test_aggregator_no_upstream()
    test_aggregator_weighted_sum()
    test_aggregator_gradient_through_weights()
    print("\nAll DAG executor tests passed!")
