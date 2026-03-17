"""
Pipeline walkthrough: trace a single question through the full 5-agent DAG
using a real Qwen3 model.

This is NOT a training test. It verifies:
1. Real model loads and runs correctly
2. Each non-terminal agent generates readable reasoning text
3. Compressor produces valid prefixes from hidden trajectories
4. Aggregator correctly combines multi-path prefixes
5. Terminal agent (summarizer) produces logits conditioned on upstream prefixes
6. The full pipeline runs without errors on a real question

Usage:
    python tests/test_pipeline_walkthrough.py

    # Or with a custom model path:
    python tests/test_pipeline_walkthrough.py --model_path /data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.base_model import BaseModelWrapper
from src.models.compressor import LatentCompressor
from src.models.agent import Agent
from src.graph.adjacency import LearnableAdjacency
from src.graph.dag_executor import DAGExecutor
from src.communication.aggregator import MessageAggregator


# ── A real GSM8K-style question ──
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
    print(f"  Vocab size: {base_model.model_config.vocab_size}")
    print(f"  Frozen: {all(not p.requires_grad for p in base_model.model.parameters())}")

    # ══════════════════════════════════════════
    #  Step 2: Create agents
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 2: Creating 5 agents")
    print("=" * 60)

    role_configs = {
        "reader": {
            "role_name": "reader",
            "system_prompt": "You are a reader agent. Carefully read the problem, identify key information, constraints, and what is being asked. Restate the problem clearly.",
            "reasoning_steps": 64,
            "compress_last_k": 20,
        },
        "planner": {
            "role_name": "planner",
            "system_prompt": "You are a planning agent. Break down the problem into clear, logical steps. Identify what information is needed and outline a solution strategy.",
            "reasoning_steps": 64,
            "compress_last_k": 20,
        },
        "analyst": {
            "role_name": "analyst",
            "system_prompt": "You are an analyst agent. Examine the problem from a high level. Identify the type of problem, relevant concepts, and possible solution strategies.",
            "reasoning_steps": 64,
            "compress_last_k": 20,
        },
        "solver": {
            "role_name": "solver",
            "system_prompt": "You are a solver agent. Based on the analysis provided, execute the solution step by step. Show your work and arrive at a concrete answer.",
            "reasoning_steps": 64,
            "compress_last_k": 20,
        },
        "summarizer": {
            "role_name": "summarizer",
            "system_prompt": "You are a summarizer agent. Synthesize the reasoning from all upstream agents, resolve any conflicts, and produce the final answer. For math problems, show the final numeric answer after ####.",
            "reasoning_steps": 64,
            "compress_last_k": 20,
        },
    }

    agent_order = ["reader", "planner", "analyst", "solver", "summarizer"]
    agents = []
    for i, name in enumerate(agent_order):
        agent = Agent(
            agent_id=i,
            role_config=role_configs[name],
            base_model=base_model,
        )
        agents.append(agent)
        print(f"  Agent {i}: {name} (reasoning_steps={agent.reasoning_steps}, compress_last_k={agent.compress_last_k})")

    # ══════════════════════════════════════════
    #  Step 3: Create compressor + adjacency
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 3: Creating compressor + adjacency")
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
    #  Step 4: Tokenize the question
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 4: Tokenizing question")
    print("=" * 60)
    print(f"  Q: {TEST_QUESTION}")
    print(f"  A: {GROUND_TRUTH}")

    tokenized = base_model.tokenize([TEST_QUESTION], max_length=128)
    task_ids = tokenized["input_ids"].to(device)
    task_mask = tokenized["attention_mask"].to(device)
    print(f"  Token IDs shape: {task_ids.shape}")
    print(f"  Tokens: {base_model.tokenizer.decode(task_ids[0])}")

    # ══════════════════════════════════════════
    #  Step 5: Walk through DAG manually
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 5: Walking through the DAG")
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
            # ── Non-terminal: generate reasoning ──
            print(f"  Generating {agent.reasoning_steps} reasoning tokens...")
            agent_output = agent.reason(
                task_token_ids=task_ids,
                task_attention_mask=task_mask,
                upstream_prefix=upstream_prefix,
            )

            # Show what the agent generated
            gen_ids = agent_output["generated_ids"]
            gen_text = base_model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            print(f"  Generated text ({gen_ids.shape[1]} tokens):")
            print(f"    \"{gen_text[:200]}{'...' if len(gen_text) > 200 else ''}\"")

            # Compress
            trajectory = agent_output["hidden_trajectory"]
            mask = agent_output["compressor_mask"]
            print(f"  Hidden trajectory for compression: shape={trajectory.shape}")

            P_j = compressor(trajectory, mask=mask)
            print(f"  Compressed prefix: shape={P_j.shape}, norm={P_j.norm().item():.4f}")
            all_prefixes.append(P_j)

        else:
            # ── Terminal: forward for logits ──
            print(f"  Terminal agent: running forward_for_loss...")
            terminal_output = agent.forward_for_loss(
                task_token_ids=task_ids,
                task_attention_mask=task_mask,
                upstream_prefix=upstream_prefix,
            )
            final_logits = terminal_output["logits"]
            print(f"  Output logits: shape={final_logits.shape}")

            # Greedy decode to see what the terminal agent would produce
            predicted_ids = final_logits.argmax(dim=-1)
            predicted_text = base_model.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            print(f"  Predicted text (from logits):")
            print(f"    \"{predicted_text[:200]}{'...' if len(predicted_text) > 200 else ''}\"")
            all_prefixes.append(None)

    # ══════════════════════════════════════════
    #  Step 6: Summary
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Step 6: Pipeline summary")
    print("=" * 60)

    print(f"\n  Graph topology:")
    print(f"    reader ──→ planner ──→ solver ──┐")
    print(f"      │                              ├──→ summarizer")
    print(f"      └────→ analyst ───────────────┘")

    print(f"\n  Information flow:")
    for j, name in enumerate(agent_order):
        upstream_names = [agent_order[i] for i in range(j) if A[i, j].item() > 0.01]
        if upstream_names:
            weights = [f"{A[agent_order.index(u), j].item():.3f}" for u in upstream_names]
            upstream_str = ", ".join(f"{n}({w})" for n, w in zip(upstream_names, weights))
            print(f"    {name}: receives from [{upstream_str}]")
        else:
            print(f"    {name}: source node (no upstream)")

    print(f"\n  Prefix shapes:")
    for j, (name, prefix) in enumerate(zip(agent_order, all_prefixes)):
        if prefix is not None:
            print(f"    {name}: {prefix.shape} (norm={prefix.norm().item():.4f})")
        else:
            print(f"    {name}: terminal (no prefix produced)")

    print(f"\n  Question: {TEST_QUESTION}")
    print(f"  Expected: {GROUND_TRUTH}")
    print(f"  Note: Without training, the summarizer cannot read latent prefixes,")
    print(f"        so the output will be garbage. This is expected.")

    print("\n" + "=" * 60)
    print("  PIPELINE WALKTHROUGH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B",
        help="Path to local Qwen3 model weights",
    )
    args = parser.parse_args()
    run_walkthrough(args.model_path)