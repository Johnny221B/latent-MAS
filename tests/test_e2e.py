"""
End-to-end smoke test for the multi-agent latent communication framework.

This test uses a TINY randomly initialized model (not real weights) to verify:
1. All modules import correctly
2. DAG execution runs without errors
3. Non-terminal agents: generation → compress → pass prefix
4. Terminal agent: forward → logits for CE loss
5. Loss computation (task + graph + sparse) produces a scalar
6. backward() runs and gradients flow to compressor + adjacency only
7. Aggregator uses direct weighted sums and still preserves gradient flow

Usage:
    python tests/test_e2e.py

If this passes, the framework is structurally correct.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


def make_tiny_config():
    """Create a minimal experiment config that doesn't need real model files."""
    import tempfile, json, yaml, os

    tmpdir = tempfile.mkdtemp()

    # ── Role configs ──
    roles_dir = os.path.join(tmpdir, "roles")
    os.makedirs(roles_dir)

    roles = {
        "reader": {
            "role_name": "reader",
            "system_prompt": "Read the problem.",
            "reasoning_steps": 8,
            "compress_last_k": 4,
        },
        "planner": {
            "role_name": "planner",
            "system_prompt": "Plan the solution.",
            "reasoning_steps": 8,
            "compress_last_k": 4,
        },
        "analyst": {
            "role_name": "analyst",
            "system_prompt": "Analyze the problem.",
            "reasoning_steps": 8,
            "compress_last_k": 4,
        },
        "solver": {
            "role_name": "solver",
            "system_prompt": "Solve step by step.",
            "reasoning_steps": 8,
            "compress_last_k": 4,
        },
        "summarizer": {
            "role_name": "summarizer",
            "system_prompt": "Summarize and give the final answer.",
            "reasoning_steps": 8,
            "compress_last_k": 4,
        },
    }
    for name, cfg in roles.items():
        with open(os.path.join(roles_dir, f"{name}.json"), "w") as f:
            json.dump(cfg, f)

    # ── Graph config (5-agent, two paths) ──
    graph_path = os.path.join(tmpdir, "graph.json")
    graph = {
        "agents": ["reader", "planner", "analyst", "solver", "summarizer"],
        "adjacency_prior": [
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        "terminal_agent_index": 4,
    }
    with open(graph_path, "w") as f:
        json.dump(graph, f)

    # ── Experiment config ──
    config = {
        "model": {
            "name": "dummy",
            "cache_dir": tmpdir,
            "frozen": True,
        },
        "graph": {
            "config": graph_path,
            "roles_dir": roles_dir,
        },
        "compressor": {
            "num_queries": 4,
            "num_heads": 2,
            "dropout": 0.0,
        },
        "training": {
            "task": "gsm8k",
            "batch_size": 2,
            "lr": 1e-4,
            "max_seq_len": 16,
            "lambda_add": 0.1,
            "lambda_drop": 0.5,
            "lambda_sparse": 0.05,
        },
        "reasoning": {
            "steps_per_agent": 8,
            "compress_last_k": 4,
        },
    }

    return config, tmpdir


class TinyLM(nn.Module):
    """A minimal causal LM for testing — no real weights needed."""

    def __init__(self, vocab_size=100, hidden_dim=32, num_layers=2):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_dim})()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False,
                use_cache=False, past_key_values=None, **kwargs):
        batch_size, seq_len, _ = inputs_embeds.shape
        h = inputs_embeds
        all_hidden = [h] if output_hidden_states else None

        for layer in self.layers:
            # Simulate attention: each position attends to all previous positions
            # via mean pooling. This ensures prefix positions influence text positions.
            cumsum = torch.cumsum(h, dim=1)
            counts = torch.arange(1, h.shape[1] + 1, device=h.device).float().unsqueeze(0).unsqueeze(-1)
            h_attn = cumsum / counts  # causal mean pooling: [B, S, D]
            h = torch.relu(layer(h_attn))
            if output_hidden_states:
                all_hidden.append(h)

        logits = self.lm_head(h)

        fake_kv = [
            (
                torch.zeros(batch_size, 1, seq_len, self.hidden_dim),
                torch.zeros(batch_size, 1, seq_len, self.hidden_dim),
            )
            for _ in self.layers
        ] if use_cache else None

        return type("Output", (), {
            "logits": logits,
            "hidden_states": tuple(all_hidden) if output_hidden_states else None,
            "past_key_values": fake_kv,
        })()


def test_e2e():
    """Run the full pipeline with a tiny model."""
    from src.models.base_model import BaseModelWrapper
    from src.models.compressor import LatentCompressor
    from src.models.agent import Agent
    from src.graph.adjacency import LearnableAdjacency
    from src.graph.dag_executor import DAGExecutor
    from src.communication.aggregator import MessageAggregator
    from src.losses.task_loss import TaskLoss
    from src.losses.graph_loss import GraphLoss

    print("=" * 60)
    print("  End-to-End Smoke Test")
    print("=" * 60)

    config, tmpdir = make_tiny_config()
    device = torch.device("cpu")

    # ── Step 1: Create a tiny model (bypass real model loading) ──
    print("\n[1/7] Creating tiny model...")
    vocab_size = 100
    hidden_dim = 32

    # Create a fake BaseModelWrapper by monkey-patching
    wrapper = BaseModelWrapper.__new__(BaseModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = TinyLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
    wrapper.model_config = wrapper.model.config

    # Create a simple tokenizer mock
    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        eos_token_id = 1
        pad_token_id = 0

        def batch_decode(self, token_ids, skip_special_tokens=True):
            return [f"question-{i}" for i in range(token_ids.shape[0])]

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True, enable_thinking=True):
            return messages[-1]["content"]

        def decode(self, token_ids, skip_special_tokens=True):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return " ".join(str(x) for x in token_ids)

        def __call__(self, texts, return_tensors=None, padding=False,
                     truncation=False, max_length=16, add_special_tokens=True, **kwargs):
            # Handle both single string and list of strings
            if isinstance(texts, str):
                texts = [texts]
            B = len(texts)
            seq_len = min(8, max_length)
            return {
                "input_ids": torch.randint(2, vocab_size, (B, seq_len)),
                "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
            }
    wrapper.tokenizer = FakeTokenizer()

    # Freeze
    for p in wrapper.model.parameters():
        p.requires_grad = False
    wrapper.model.eval()

    print(f"  hidden_dim={hidden_dim}, vocab={vocab_size}")
    print("  ✓ Tiny model created")

    # ── Step 2: Create agents ──
    print("\n[2/7] Creating agents...")
    import json, os

    roles_dir = config["graph"]["roles_dir"]
    with open(config["graph"]["config"]) as f:
        graph_config = json.load(f)

    agents = []
    for i, role_name in enumerate(graph_config["agents"]):
        with open(os.path.join(roles_dir, f"{role_name}.json")) as f:
            role_config = json.load(f)
        role_config["reasoning_steps"] = config["reasoning"]["steps_per_agent"]
        role_config["compress_last_k"] = config["reasoning"]["compress_last_k"]
        agent = Agent(agent_id=i, role_config=role_config, base_model=wrapper)
        agents.append(agent)
        print(f"  Agent {i}: {role_name}")
    print("  ✓ All agents created")

    # ── Step 3: Create trainable modules ──
    print("\n[3/7] Creating compressor + adjacency...")
    compressor = LatentCompressor(
        hidden_dim=hidden_dim,
        num_queries=config["compressor"]["num_queries"],
        num_heads=config["compressor"]["num_heads"],
        dropout=0.0,
    )

    prior = torch.tensor(graph_config["adjacency_prior"], dtype=torch.float32)
    adjacency = LearnableAdjacency(prior=prior)

    print(f"  Compressor: {sum(p.numel() for p in compressor.parameters())} params")
    print(f"  Adjacency: {adjacency.n}x{adjacency.n}")
    A = adjacency.get_adjacency()
    print(f"  Initial A:\n{A.detach()}")
    print("  ✓ Trainable modules created")

    # ── Step 4: Run DAG execution ──
    print("\n[4/7] Running DAG execution...")
    executor = DAGExecutor(aggregator=MessageAggregator())

    B = 2
    task_ids = torch.randint(2, vocab_size, (B, 8))
    task_mask = torch.ones(B, 8, dtype=torch.long)

    dag_output = executor.execute(
        agents=agents,
        adjacency=A,
        compressor=compressor,
        task_token_ids=task_ids,
        task_attention_mask=task_mask,
    )

    print(f"  final_logits shape: {dag_output['final_logits'].shape}")
    print(f"  num prefixes: {sum(1 for p in dag_output['all_prefixes'] if p is not None)}")
    print("  ✓ DAG execution complete")

    # ── Step 5: Compute loss ──
    print("\n[5/7] Computing loss...")
    # ── Diagnose gradient flow ──
    print("\n[5.5/7] Diagnosing gradient path...")
    # Check if prefixes have grad_fn
    for i, p in enumerate(dag_output["all_prefixes"]):
        if p is not None:
            print(f"  Prefix {i}: requires_grad={p.requires_grad}, has grad_fn={p.grad_fn is not None}")
    fl = dag_output["final_logits"]
    print(f"  final_logits: requires_grad={fl.requires_grad}, has grad_fn={fl.grad_fn is not None}")
    task_loss_fn = TaskLoss()
    graph_loss_fn = GraphLoss(lambda_add=0.1, lambda_drop=0.5, lambda_sparse=0.05)

    final_logits = dag_output["final_logits"]
    # Create fake labels matching the task portion length
    labels = torch.randint(0, vocab_size, (B, final_logits.shape[1]))

    task_loss = task_loss_fn(final_logits, labels)
    graph_loss_dict = graph_loss_fn(A, prior)
    total_loss = task_loss + graph_loss_dict["loss"]

    print(f"  task_loss:    {task_loss.item():.4f}")
    print(f"  graph_loss:   {graph_loss_dict['loss'].item():.4f}")
    print(f"    loss_add:   {graph_loss_dict['loss_add'].item():.4f}")
    print(f"    loss_drop:  {graph_loss_dict['loss_drop'].item():.4f}")
    print(f"    loss_sparse:{graph_loss_dict['loss_sparse'].item():.4f}")
    print(f"  total_loss:   {total_loss.item():.4f}")
    print("  ✓ Loss computed")

    # ── Step 6: Backward pass ──
    print("\n[6/7] Running backward...")
    total_loss.backward()

    # Check compressor gradients — detailed
    print("  Compressor parameter gradients:")
    for name, p in compressor.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.abs().sum().item()
            print(f"    {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"    {name}: grad=None")
    comp_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in compressor.parameters())
    print(f"  Compressor has gradients: {comp_has_grad}")

    # Check adjacency gradients
    adj_has_grad = (adjacency.logits.grad is not None and
                    adjacency.logits.grad.abs().sum() > 0)
    print(f"  Adjacency has gradients:  {adj_has_grad}")

    # Check frozen model has NO gradients
    frozen_has_grad = any(p.grad is not None for p in wrapper.model.parameters())
    print(f"  Frozen model has grads:   {frozen_has_grad} (should be False)")

    assert comp_has_grad, "Compressor should receive gradients!"
    assert adj_has_grad, "Adjacency should receive gradients!"
    assert not frozen_has_grad, "Frozen model should NOT receive gradients!"
    print("  ✓ Gradient flow verified")

    # ── Step 7: Verify aggregator weighted sum ──
    print("\n[7/7] Verifying aggregator weighted sum...")
    agg = MessageAggregator()
    P0 = torch.randn(2, 4, hidden_dim)
    P1 = torch.randn(2, 4, hidden_dim)
    test_adj = torch.tensor([
        [0, 0, 0, 0, 0.8],
        [0, 0, 0, 0, 0.0],
        [0, 0, 0, 0, 0.6],
        [0, 0, 0, 0, 0.9],
        [0, 0, 0, 0, 0.0],
    ])
    # Agent 4 receives from agent 0 (0.8) and agent 2 (0.6) and agent 3 (0.9)
    z = agg.aggregate(
        agent_index=4,
        adjacency=test_adj,
        all_prefixes=[P0, None, P1, P1, None],
    )
    expected = 0.8 * P0 + 0.6 * P1 + 0.9 * P1
    assert torch.allclose(z, expected, atol=1e-5), "Aggregator weighted sum failed!"
    print(f"  Weights: [0.8, 0.6, 0.9]")
    print("  Weighted sum correctly: ✓")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nPipeline summary:")
    print(f"  Agents: {len(agents)} ({', '.join(a.role_name for a in agents)})")
    print(f"  Graph: {graph_config['description'] if 'description' in graph_config else 'custom'}")
    print(f"  Trainable params: compressor({sum(p.numel() for p in compressor.parameters())}) + adjacency({adjacency.logits.numel()})")
    print(f"  Loss: {total_loss.item():.4f}")
    print(f"  Gradient flow: compressor ✓, adjacency ✓, frozen ✗")


if __name__ == "__main__":
    test_e2e()
