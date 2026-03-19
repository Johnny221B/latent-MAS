"""Tests for DAGExecutor and MessageAggregator."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.communication.aggregator import MessageAggregator
from src.graph.dag_executor import DAGExecutor


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

    # Aggregator normalizes by the total incoming weight.
    expected = (0.5 * P0 + 0.3 * P1) / (0.5 + 0.3)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected[0,0,0]}, got {result[0,0,0]}"
    print("✓ test_aggregator_weighted_sum passed")


def test_aggregator_gradient_through_weights():
    """Gradients should flow through adjacency weights."""
    agg = MessageAggregator()
    B, Lp, D = 2, 4, 8

    P0 = torch.randn(B, Lp, D)
    P1 = torch.randn(B, Lp, D)

    # Make adjacency require grad (simulating LearnableAdjacency)
    adjacency = torch.tensor([
        [0, 0, 0.8],
        [0, 0, 0.2],
        [0, 0, 0],
    ], requires_grad=True)

    result = agg.aggregate(agent_index=2, adjacency=adjacency, all_prefixes=[P0, P1])
    loss = result.sum()
    loss.backward()

    assert adjacency.grad is not None, "Adjacency should receive gradients"
    assert adjacency.grad[0, 2].item() != 0, "Edge weight gradient should be non-zero"
    print("✓ test_aggregator_gradient_through_weights passed")


def test_dag_executor_returns_generation_metadata():
    class FakeAgent:
        def reason(self, **kwargs):
            return {
                "hidden_trajectory": torch.zeros(1, 1, 4),
                "compressor_mask": torch.ones(1, 1),
            }

        def generate_answer(self, **kwargs):
            assert kwargs["return_metadata"] is True
            return {
                "generated_text": "42",
                "finish_reason": "eos",
                "generated_token_count": 3,
                "stopped_early": True,
            }

    class FakeCompressor:
        def __call__(self, hidden, mask=None):
            return hidden

    executor = DAGExecutor(aggregator=MessageAggregator())
    adjacency = torch.zeros(2, 2)
    out = executor.execute(
        agents=[FakeAgent(), FakeAgent()],
        adjacency=adjacency,
        compressor=FakeCompressor(),
        task_token_ids=torch.ones(1, 2, dtype=torch.long),
        training=False,
    )
    assert out["generated_text"] == "42"
    assert out["generation"]["finish_reason"] == "eos"


def test_dag_executor_passes_max_new_tokens_to_terminal_agent():
    class FakeAgent:
        def __init__(self):
            self.seen_max_new_tokens = None

        def reason(self, **kwargs):
            return {
                "hidden_trajectory": torch.zeros(1, 1, 4),
                "compressor_mask": torch.ones(1, 1),
            }

        def generate_answer(self, **kwargs):
            self.seen_max_new_tokens = kwargs["max_new_tokens"]
            return {
                "generated_text": "42",
                "finish_reason": "max_new_tokens",
                "generated_token_count": kwargs["max_new_tokens"],
                "stopped_early": False,
            }

    class FakeCompressor:
        def __call__(self, hidden, mask=None):
            return hidden

    first = FakeAgent()
    terminal = FakeAgent()
    executor = DAGExecutor(aggregator=MessageAggregator())
    adjacency = torch.zeros(2, 2)

    out = executor.execute(
        agents=[first, terminal],
        adjacency=adjacency,
        compressor=FakeCompressor(),
        task_token_ids=torch.ones(1, 2, dtype=torch.long),
        training=False,
        max_new_tokens=2048,
    )

    assert terminal.seen_max_new_tokens == 2048
    assert out["generation"]["generated_token_count"] == 2048


def test_dag_executor_passes_terminal_inference_options():
    class FakeAgent:
        def __init__(self):
            self.kwargs = None

        def reason(self, **kwargs):
            return {
                "hidden_trajectory": torch.zeros(1, 1, 4),
                "compressor_mask": torch.ones(1, 1),
            }

        def generate_answer(self, **kwargs):
            self.kwargs = kwargs
            return {
                "generated_text": "42",
                "finish_reason": "eos",
                "generated_token_count": 2,
                "stopped_early": True,
            }

    class FakeCompressor:
        def __call__(self, hidden, mask=None):
            return hidden

    first = FakeAgent()
    terminal = FakeAgent()
    executor = DAGExecutor(aggregator=MessageAggregator())
    adjacency = torch.zeros(2, 2)

    executor.execute(
        agents=[first, terminal],
        adjacency=adjacency,
        compressor=FakeCompressor(),
        task_token_ids=torch.ones(1, 2, dtype=torch.long),
        training=False,
        max_new_tokens=128,
        inference_mode="chat_with_prefix",
        use_terminal_prefix=False,
        do_sample=False,
    )

    assert terminal.kwargs["inference_mode"] == "chat_with_prefix"
    assert terminal.kwargs["use_upstream_prefix"] is False
    assert terminal.kwargs["do_sample"] is False


if __name__ == "__main__":
    test_aggregator_no_upstream()
    test_aggregator_weighted_sum()
    test_aggregator_gradient_through_weights()
    print("\nAll DAG executor tests passed!")
