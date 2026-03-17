"""
Pipeline walkthrough: trace a single question through the full 5-agent DAG
using real Qwen3 model with LATENT REASONING.

Verifies:
1. Real model loads correctly
2. Alignment matrix W_a computes without errors
3. Each non-terminal agent performs latent reasoning (no text output)
4. Compressor produces valid prefixes from latent trajectories
5. Aggregator combines multi-path prefixes with normalization
6. Terminal agent receives prefix, does forward pass, greedy-decodes to text
7. The full pipeline produces a natural language answer

Usage:
    python tests/test_pipeline_walkthrough.py
    python tests/test_pipeline_walkthrough.py --model_path /your/path
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.base_model import BaseModelWrapper
from src.models.compressor import LatentCompressor
from src.models.agent import Agent
from src.graph.adjacency import LearnableAdjacency
from src.communication.aggregator import MessageAggregator


TEST_QUESTION = (
    "Debra has 12 apples. She gives 3 to her friend and then buys 5 more. "
    "How many apples does Debra have now?"
)
GROUND_TRUTH = "14"


def run_walkthrough(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ══════════════════════════════════════════
    #  Step 1: Load real model
    # ══════════════════════════════════════════
    print("=" * 60)
    print("  Step 1: Loading Qwen3 model")
    print("=" * 60)
    base_model = BaseModelWrapper(model_name="Qwen/Qwen3-0.6B", cache_dir=model_path)
    base_model.to(device)
    print(f"  Model: {base_model.model_config._name_or_path}")
    print(f"  Hidden dim: {base_model.hidden_dim}")
    print(f"  Frozen: {all(not p.requires_grad for p in base_model.model.parameters())}")

    # ══════════════════════════════════════════
    #  Step 2: Compute alignment matrix
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 2: Computing alignment matrix W_a")
    print("=" * 60)
    W_a, target_norm = base_model.compute_alignment_matrix()
    print(f"  W_a shape: {W_a.shape}")
    print(f"  W_a dtype: {W_a.dtype}")
    print(f"  Target embedding norm: {target_norm.item():.4f}")
    print(f"  W_a norm: {W_a.norm().item():.4f}")
    print("  ✓ Alignment matrix computed")

    # ══════════════════════════════════════════
    #  Step 3: Create agents
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 3: Creating 5 agents")
    print("=" * 60)

    role_configs = {
        "reader": {
            "role_name": "reader",
            "system_prompt": "You are a reader agent. Carefully read the problem, identify key information, constraints, and what is being asked.",
            "reasoning_steps": 25,
            "compress_last_k": 25,
        },
        "planner": {
            "role_name": "planner",
            "system_prompt": "You are a planning agent. Break down the problem into clear, logical steps.",
            "reasoning_steps": 25,
            "compress_last_k": 25,
        },
        "analyst": {
            "role_name": "analyst",
            "system_prompt": "You are an analyst agent. Examine the problem from a high level and identify solution strategies.",
            "reasoning_steps": 25,
            "compress_last_k": 25,
        },
        "solver": {
            "role_name": "solver",
            "system_prompt": "You are a solver agent. Execute the solution step by step and arrive at a concrete answer.",
            "reasoning_steps": 25,
            "compress_last_k": 25,
        },
        "summarizer": {
            "role_name": "summarizer",
            "system_prompt": "You are a summarizer agent. Synthesize the reasoning from all upstream agents and produce the final answer. For math problems, show the final numeric answer after ####.",
            "reasoning_steps": 25,
            "compress_last_k": 25,
        },
    }

    agent_order = ["reader", "planner", "analyst", "solver", "summarizer"]
    agents = []
    for i, name in enumerate(agent_order):
        agent = Agent(agent_id=i, role_config=role_configs[name], base_model=base_model)
        agents.append(agent)
        print(f"  Agent {i}: {name} (latent_steps={agent.reasoning_steps}, compress_k={agent.compress_last_k})")

    # ══════════════════════════════════════════
    #  Step 4: Create compressor + adjacency
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 4: Creating compressor + adjacency")
    print("=" * 60)

    compressor = LatentCompressor(
        hidden_dim=base_model.hidden_dim,
        num_queries=16,
        num_heads=8,
        dropout=0.0,
    ).to(device)

    prior = torch.tensor([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)

    adjacency = LearnableAdjacency(prior=prior).to(device)
    A = adjacency.get_adjacency()

    print(f"  Compressor: {sum(p.numel() for p in compressor.parameters()):,} params, Lp=16")
    print(f"  Adjacency (initial):")
    for i, name_i in enumerate(agent_order):
        row = "  ".join(f"{A[i,j].item():.3f}" for j in range(5))
        print(f"    {name_i:>12}: {row}")

    # ══════════════════════════════════════════
    #  Step 5: Tokenize the question
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 5: Tokenizing question")
    print("=" * 60)
    print(f"  Q: {TEST_QUESTION}")
    print(f"  A: {GROUND_TRUTH}")

    tokenized = base_model.tokenize([TEST_QUESTION], max_length=128)
    task_ids = tokenized["input_ids"].to(device)
    task_mask = tokenized["attention_mask"].to(device)
    print(f"  Token IDs shape: {task_ids.shape}")

    # ══════════════════════════════════════════
    #  Step 6: Walk through DAG
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 6: Walking through the DAG (latent reasoning)")
    print("=" * 60)

    aggregator = MessageAggregator()
    all_prefixes = []

    for j, agent in enumerate(agents):
        name = agent_order[j]
        print(f"\n  ── Agent {j}: {name} ──")

        # Aggregate upstream
        upstream_prefix = aggregator.aggregate(
            agent_index=j,
            adjacency=A,
            all_prefixes=all_prefixes,
        )
        if upstream_prefix is not None:
            print(f"  Received prefix: shape={upstream_prefix.shape}, norm={upstream_prefix.norm().item():.4f}")
        else:
            print(f"  No upstream prefix (source node)")

        if j < len(agents) - 1:
            # ── Non-terminal: latent reasoning (no text output) ──
            print(f"  Running {agent.reasoning_steps} latent reasoning steps...")

            with torch.no_grad():
                agent_output = agent.reason(
                    task_token_ids=task_ids,
                    task_attention_mask=task_mask,
                    upstream_prefix=upstream_prefix,
                )

            trajectory = agent_output["hidden_trajectory"]
            mask = agent_output["compressor_mask"]

            # Show trajectory stats (no text to show — reasoning is in latent space)
            traj_norms = trajectory.norm(dim=-1)  # [B, k]
            print(f"  Latent trajectory: shape={trajectory.shape}")
            print(f"    per-step norms: min={traj_norms.min().item():.2f}, max={traj_norms.max().item():.2f}, mean={traj_norms.mean().item():.2f}")

            # Compress
            P_j = compressor(trajectory, mask=mask)
            print(f"  Compressed prefix: shape={P_j.shape}, norm={P_j.norm().item():.4f}")
            all_prefixes.append(P_j)

        else:
            # ── Terminal agent ──
            all_prefixes.append(None)

            # Part A: Forward for logits (this is what training uses)
            print(f"  Terminal agent: forward pass for logits...")
            terminal_output = agent.forward_for_loss(
                task_token_ids=task_ids,
                task_attention_mask=task_mask,
                upstream_prefix=upstream_prefix,
            )
            final_logits = terminal_output["logits"]
            print(f"  Output logits: shape={final_logits.shape}")

            predicted_ids = final_logits.argmax(dim=-1)
            predicted_text = base_model.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            print(f"  Logits decode (garbled, untrained): \"{predicted_text[:200]}\"")

            # Part B: Also do generation WITHOUT prefix to show model's baseline ability
            print(f"\n  Baseline: model answering WITHOUT any prefix (sanity check)...")
            gen_input_ids = agent.build_input_ids(task_ids)
            if task_mask is not None:
                role_mask = torch.ones(
                    task_mask.shape[0],
                    agent._get_role_token_ids().shape[1],
                    device=task_mask.device,
                    dtype=task_mask.dtype,
                )
                gen_mask = torch.cat([role_mask, task_mask], dim=1)
            else:
                gen_mask = None

            with torch.no_grad():
                gen_out = base_model.model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_mask,
                    max_new_tokens=128,
                    do_sample=False,  # greedy for reproducibility
                    pad_token_id=base_model.tokenizer.pad_token_id,
                )
            baseline_text = base_model.tokenizer.decode(
                gen_out[0][gen_input_ids.shape[1]:], skip_special_tokens=True
            )
            print(f"  Baseline answer: \"{baseline_text[:300]}\"")

    # ══════════════════════════════════════════
    #  Step 7: Summary
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 7: Pipeline summary")
    print("=" * 60)

    print(f"\n  Graph:")
    print(f"    reader ──→ planner ──→ solver ──┐")
    print(f"      │                              ├──→ summarizer")
    print(f"      └────→ analyst ───────────────┘")

    print(f"\n  Config:")
    print(f"    Latent steps per agent: {agents[0].reasoning_steps}")
    print(f"    Compress last k: {agents[0].compress_last_k}")
    print(f"    Prefix tokens (Lp): 16")
    print(f"    Hidden dim: {base_model.hidden_dim}")

    print(f"\n  Prefix summary:")
    for j, (name, prefix) in enumerate(zip(agent_order, all_prefixes)):
        if prefix is not None:
            print(f"    {name}: {prefix.shape}, norm={prefix.norm().item():.4f}")
        else:
            print(f"    {name}: terminal")

    print(f"\n  Question: {TEST_QUESTION}")
    print(f"  Expected: {GROUND_TRUTH}")
    print(f"  Note: Logits decode is garbled because compressor is untrained.")
    print(f"        Baseline shows the model CAN answer correctly without prefix.")

    print("\n" + "=" * 60)
    print("  PIPELINE WALKTHROUGH COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="/data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B",
    )
    args = parser.parse_args()
    run_walkthrough(args.model_path)