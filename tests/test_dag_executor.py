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

    expected = 0.5 * P0 + 0.3 * P1
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


def test_dag_executor_collects_agent_logs():
    class FakeAgent:
        def __init__(self, agent_id, role_name):
            self.agent_id = agent_id
            self.role_name = role_name
            self.system_prompt = f"prompt-{role_name}"

        def reason(self, **kwargs):
            return {
                "hidden_trajectory": torch.ones(1, 2, 4),
                "compressor_mask": torch.ones(1, 2),
            }

        def generate_answer(self, **kwargs):
            return {
                "generated_text": "42",
                "finish_reason": "eos",
                "generated_token_count": 2,
                "stopped_early": True,
            }

    class FakeCompressor:
        def __call__(self, hidden, mask=None):
            return torch.ones(1, 1, 4) * 3

    executor = DAGExecutor(aggregator=MessageAggregator())
    out = executor.execute(
        agents=[FakeAgent(0, "reader"), FakeAgent(1, "solver")],
        adjacency=torch.zeros(2, 2),
        compressor=FakeCompressor(),
        task_token_ids=torch.ones(1, 2, dtype=torch.long),
        training=False,
        collect_agent_logs=True,
    )

    assert len(out["agent_logs"]) == 2
    assert out["agent_logs"][0]["output_type"] == "latent"
    assert out["agent_logs"][0]["hidden_trajectory"]["shape"] == [1, 2, 4]
    assert out["agent_logs"][1]["output_type"] == "text"
    assert out["agent_logs"][1]["generated_text"] == "42"


def test_dag_executor_respects_explicit_execution_order_for_non_terminal_agents():
    call_order = []

    class FakeAgent:
        def __init__(self, agent_id, role_name):
            self.agent_id = agent_id
            self.role_name = role_name
            self.system_prompt = role_name
            self.received_prefix = None

        def reason(self, **kwargs):
            call_order.append(self.role_name)
            self.received_prefix = kwargs["upstream_prefix"]
            return {
                "hidden_trajectory": torch.full((1, 1, 4), float(self.agent_id + 1)),
                "compressor_mask": torch.ones(1, 1),
            }

        def generate_answer(self, **kwargs):
            call_order.append(self.role_name)
            self.received_prefix = kwargs["upstream_prefix"]
            return {
                "generated_text": "done",
                "finish_reason": "eos",
                "generated_token_count": 1,
                "stopped_early": True,
            }

    class FakeCompressor:
        def __call__(self, hidden, mask=None):
            return hidden

    agents = [
        FakeAgent(0, "reader"),
        FakeAgent(1, "summarizer"),
        FakeAgent(2, "solver"),
    ]
    adjacency = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    executor = DAGExecutor(aggregator=MessageAggregator())
    out = executor.execute(
        agents=agents,
        adjacency=adjacency,
        compressor=FakeCompressor(),
        task_token_ids=torch.ones(1, 2, dtype=torch.long),
        training=False,
        execution_order=[0, 2, 1],
        terminal_agent_index=1,
    )

    assert out["generated_text"] == "done"
    assert call_order == ["reader", "solver", "summarizer"]
    assert agents[1].received_prefix is not None
    assert torch.allclose(agents[1].received_prefix, torch.full((1, 1, 4), 3.0))


def test_dag_executor_text_messages_follow_dag_routes():
    class FakeAgent:
        def __init__(self, agent_id, role_name):
            self.agent_id = agent_id
            self.role_name = role_name
            self.system_prompt = role_name
            self.upstream_text_messages = None

        def generate_message(self, **kwargs):
            self.upstream_text_messages = kwargs["upstream_text_messages"]
            return {
                "generated_text": f"msg-{self.role_name}",
                "finish_reason": "eos",
                "generated_token_count": 3,
                "stopped_early": True,
            }

        def generate_answer(self, **kwargs):
            self.upstream_text_messages = kwargs["upstream_text_messages"]
            self.use_upstream_prefix = kwargs["use_upstream_prefix"]
            return {
                "generated_text": "final-answer",
                "finish_reason": "eos",
                "generated_token_count": 4,
                "stopped_early": True,
            }

    class FakeCompressor:
        def __call__(self, hidden, mask=None):
            raise AssertionError("compressor should not be used in text_messages mode")

    agents = [
        FakeAgent(0, "reader"),
        FakeAgent(1, "planner"),
        FakeAgent(2, "solver"),
    ]
    adjacency = torch.tensor([
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])

    executor = DAGExecutor(aggregator=MessageAggregator())
    out = executor.execute(
        agents=agents,
        adjacency=adjacency,
        compressor=FakeCompressor(),
        task_token_ids=torch.ones(1, 2, dtype=torch.long),
        training=False,
        inference_mode="chat_with_prefix",
        communication_mode="text_messages",
        collect_agent_logs=True,
    )

    assert out["generated_text"] == "final-answer"
    assert agents[0].upstream_text_messages == []
    assert agents[1].upstream_text_messages == [
        {"agent_id": 0, "role_name": "reader", "content": "msg-reader", "edge_weight": 1.0}
    ]
    assert agents[2].upstream_text_messages == [
        {"agent_id": 0, "role_name": "reader", "content": "msg-reader", "edge_weight": 1.0},
        {"agent_id": 1, "role_name": "planner", "content": "msg-planner", "edge_weight": 1.0},
    ]
    assert agents[2].use_upstream_prefix is False
    assert out["agent_logs"][0]["output_type"] == "text_message"
    assert out["agent_logs"][1]["output_type"] == "text_message"


def test_dag_executor_text_messages_preserve_per_sample_batches():
    class FakeAgent:
        def __init__(self, agent_id, role_name):
            self.agent_id = agent_id
            self.role_name = role_name
            self.system_prompt = role_name
            self.seen_messages = None

        def generate_message(self, **kwargs):
            self.seen_messages = kwargs["upstream_text_messages"]
            if self.agent_id == 0:
                return {
                    "generated_text": ["reader-0", "reader-1"],
                    "finish_reason": ["eos", "eos"],
                    "generated_token_count": [2, 2],
                    "stopped_early": [True, True],
                }
            return {
                "generated_text": ["planner-0", "planner-1"],
                "finish_reason": ["eos", "eos"],
                "generated_token_count": [2, 2],
                "stopped_early": [True, True],
            }

        def generate_answer(self, **kwargs):
            self.seen_messages = kwargs["upstream_text_messages"]
            return {
                "generated_text": ["ans-0", "ans-1"],
                "finish_reason": ["eos", "eos"],
                "generated_token_count": [2, 2],
                "stopped_early": [True, True],
            }

    class FakeCompressor:
        def __call__(self, hidden, mask=None):
            raise AssertionError("compressor should not be used in text_messages mode")

    agents = [
        FakeAgent(0, "reader"),
        FakeAgent(1, "planner"),
        FakeAgent(2, "solver"),
    ]
    adjacency = torch.tensor([
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])

    executor = DAGExecutor(aggregator=MessageAggregator())
    out = executor.execute(
        agents=agents,
        adjacency=adjacency,
        compressor=FakeCompressor(),
        task_token_ids=torch.ones(2, 2, dtype=torch.long),
        training=False,
        inference_mode="chat_with_prefix",
        communication_mode="text_messages",
    )

    assert out["generated_text"] == ["ans-0", "ans-1"]
    assert agents[1].seen_messages == [
        [{"agent_id": 0, "role_name": "reader", "content": "reader-0", "edge_weight": 1.0}],
        [{"agent_id": 0, "role_name": "reader", "content": "reader-1", "edge_weight": 1.0}],
    ]
    assert agents[2].seen_messages == [
        [
            {"agent_id": 0, "role_name": "reader", "content": "reader-0", "edge_weight": 1.0},
            {"agent_id": 1, "role_name": "planner", "content": "planner-0", "edge_weight": 1.0},
        ],
        [
            {"agent_id": 0, "role_name": "reader", "content": "reader-1", "edge_weight": 1.0},
            {"agent_id": 1, "role_name": "planner", "content": "planner-1", "edge_weight": 1.0},
        ],
    ]


if __name__ == "__main__":
    test_aggregator_no_upstream()
    test_aggregator_weighted_sum()
    test_aggregator_gradient_through_weights()
    print("\nAll DAG executor tests passed!")
