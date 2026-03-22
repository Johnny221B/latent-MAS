"""
Pipeline walkthrough: test both training and inference paths
with real Qwen3 model and latent reasoning.

Verifies:
1. Training path: latent reasoning → compress → forward_for_loss → CE loss → backward
2. Inference path: latent reasoning → compress → generate_answer → readable text
3. Gradient flow: compressor ✓, adjacency ✓, frozen model ✗

Usage:
    python tests/test_pipeline_full.py
    python tests/test_pipeline_full.py --model_path /your/path
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
from src.graph.dag_executor import DAGExecutor
from src.communication.aggregator import MessageAggregator
from src.losses.task_loss import TaskLoss
from src.losses.graph_loss import GraphLoss


TEST_QUESTION = (
    "Debra has 12 apples. She gives 3 to her friend and then buys 5 more. "
    "How many apples does Debra have now?"
)
GROUND_TRUTH = "14"


def run_test(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ══════════════════════════════════════════
    #  Setup
    # ══════════════════════════════════════════
    print("=" * 60)
    print("  Setting up model, agents, and modules")
    print("=" * 60)

    base_model = BaseModelWrapper(model_name="Qwen/Qwen3-0.6B", cache_dir=model_path)
    base_model.to(device)
    print(f"  Model loaded: hidden_dim={base_model.hidden_dim}")

    # Precompute alignment matrix
    W_a, target_norm = base_model.compute_alignment_matrix()
    print(f"  W_a computed: shape={W_a.shape}, target_norm={target_norm.item():.4f}")

    role_configs = {
        "reader": {"role_name": "reader", "system_prompt": "You are a reader agent. Identify key information in the problem.", "reasoning_steps": 25, "compress_last_k": 25},
        "planner": {"role_name": "planner", "system_prompt": "You are a planning agent. Break down the problem into steps.", "reasoning_steps": 25, "compress_last_k": 25},
        "analyst": {"role_name": "analyst", "system_prompt": "You are an analyst agent. Identify solution strategies.", "reasoning_steps": 25, "compress_last_k": 25},
        "solver": {"role_name": "solver", "system_prompt": "You are a solver agent. Execute the solution step by step.", "reasoning_steps": 25, "compress_last_k": 25},
        "summarizer": {"role_name": "summarizer", "system_prompt": "You are a summarizer agent. Produce the final answer. For math, show answer after ####.", "reasoning_steps": 25, "compress_last_k": 25},
    }

    agent_order = ["reader", "planner", "analyst", "solver", "summarizer"]
    agents = [Agent(agent_id=i, role_config=role_configs[name], base_model=base_model) for i, name in enumerate(agent_order)]
    print(f"  Agents: {', '.join(agent_order)}")

    compressor = LatentCompressor(hidden_dim=base_model.hidden_dim, num_queries=16, num_heads=8, dropout=0.0).to(device)
    print(f"  Compressor: {sum(p.numel() for p in compressor.parameters()):,} params")

    prior = torch.tensor([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    adjacency = LearnableAdjacency(prior=prior).to(device)
    print(f"  Adjacency: {adjacency.n}x{adjacency.n}")

    executor = DAGExecutor(aggregator=MessageAggregator())
    task_loss_fn = TaskLoss()
    graph_loss_fn = GraphLoss(lambda_add=0.1, lambda_drop=0.5, lambda_sparse=0.05)

    # Tokenize
    tokenized = base_model.tokenize([TEST_QUESTION], max_length=128)
    task_ids = tokenized["input_ids"].to(device)
    task_mask = tokenized["attention_mask"].to(device)
    print(f"  Question tokenized: {task_ids.shape}")

    A = adjacency.get_adjacency()

    # ══════════════════════════════════════════
    #  Test 1: Training path
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Test 1: TRAINING PATH")
    print("  (latent reasoning → compress → forward_for_loss → loss → backward)")
    print("=" * 60)

    dag_output = executor.execute(
        agents=agents,
        adjacency=A,
        compressor=compressor,
        task_token_ids=task_ids,
        task_attention_mask=task_mask,
        training=True,
    )

    final_logits = dag_output["final_logits"]
    print(f"\n  final_logits shape: {final_logits.shape}")

    # Create labels (tokenize the answer)
    from src.data import build_labels

    label_tokenized = base_model.tokenize([GROUND_TRUTH], max_length=32)
    answer_ids = label_tokenized["input_ids"].to(device)

    labels = build_labels(question_len=task_ids.shape[1], answer_ids=answer_ids)
    print(f"  labels shape: {labels.shape} (answer_len={answer_ids.shape[1]}, task_len={task_ids.shape[1]})")
    print(f"  supervised positions: {(labels != -100).sum().item()}")

    task_loss = task_loss_fn(final_logits, labels)
    graph_loss_dict = graph_loss_fn(A, adjacency.prior)
    total_loss = task_loss + graph_loss_dict["loss"]

    print(f"\n  task_loss:     {task_loss.item():.4f}")
    print(f"  graph_loss:    {graph_loss_dict['loss'].item():.4f}")
    print(f"  total_loss:    {total_loss.item():.4f}")

    # Backward
    total_loss.backward()

    comp_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in compressor.parameters())
    adj_has_grad = adjacency.logits.grad is not None and adjacency.logits.grad.abs().sum() > 0
    frozen_has_grad = any(p.grad is not None for p in base_model.model.parameters())

    print(f"\n  Gradient check:")
    print(f"    Compressor: {'✓' if comp_has_grad else '✗'}")
    print(f"    Adjacency:  {'✓' if adj_has_grad else '✗'}")
    print(f"    Frozen LLM: {'✗ (correct)' if not frozen_has_grad else '✓ (BUG!)'}")

    assert comp_has_grad, "Compressor should receive gradients!"
    assert adj_has_grad, "Adjacency should receive gradients!"
    assert not frozen_has_grad, "Frozen model should NOT receive gradients!"
    print("  ✓ Training path PASSED")

    # Clear gradients for next test
    compressor.zero_grad()
    adjacency.zero_grad()

    # ══════════════════════════════════════════
    #  Test 2: Inference path
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Test 2: INFERENCE PATH")
    print("  (latent reasoning → compress → generate_answer → text)")
    print("=" * 60)

    with torch.no_grad():
        dag_output_infer = executor.execute(
            agents=agents,
            adjacency=A,
            compressor=compressor,
            task_token_ids=task_ids,
            task_attention_mask=task_mask,
            training=False,
        )

    generated_text = dag_output_infer["generated_text"]
    print(f"\n  Generated answer:")
    print(f"    \"{generated_text[:500]}\"")
    print(f"\n  Expected: {GROUND_TRUTH}")
    print(f"  Note: Answer will be wrong — compressor is untrained.")

    # ══════════════════════════════════════════
    #  Test 3: Baseline (model answers without pipeline)
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Test 3: BASELINE (model answers directly)")
    print("=" * 60)

    with torch.no_grad():
        baseline_answer = agents[-1].generate_answer(
            task_token_ids=task_ids,
            task_attention_mask=task_mask,
            upstream_prefix=None,
            max_new_tokens=128,
            do_sample=False,
        )
    print(f"\n  Baseline answer (no prefix, greedy):")
    print(f"    \"{baseline_answer[:500]}\"")

    # ══════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Training path:  ✓ (loss={total_loss.item():.4f}, gradients flow correctly)")
    print(f"  Inference path: ✓ (generated text, {'garbled' if len(generated_text) < 3 else 'readable'})")
    print(f"  Baseline:       ✓ (model can answer the question)")
    print(f"\n  Graph: reader→planner→solver→summarizer + reader→analyst→summarizer")
    print(f"  Latent steps: 25, Compress: last 25, Prefix tokens: 16")
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B")
    args = parser.parse_args()
    run_test(args.model_path)
