"""
DAGExecutor: orchestrates agent execution in topological order.

Responsibilities:
  - Determine execution order (topological sort of the DAG)
  - For each agent: aggregate upstream prefixes, run reasoning, compress output
  - Return the final agent's output for loss computation

Since we enforce topological_order = agent_index (via upper-triangular adjacency),
the execution order is simply [0, 1, ..., n-1].

Extensibility:
  - Multi-round: wrap execute() in a loop with convergence check.
  - Dynamic DAG: recompute topological order if the graph structure changes.
"""

import torch

from ..models.agent import Agent
from ..models.compressor import LatentCompressor
from ..communication.aggregator import MessageAggregator


class DAGExecutor:
    """Executes the multi-agent DAG in topological order."""

    def __init__(self, aggregator: MessageAggregator | None = None):
        """
        Args:
            aggregator: MessageAggregator instance. If None, creates default.
        """
        self.aggregator = aggregator or MessageAggregator()

    def execute(
        self,
        agents: list[Agent],
        adjacency: torch.Tensor,
        compressor: LatentCompressor,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """Execute all agents in topological order.

        Args:
            agents: list of Agent instances, ordered by index (= topological order)
            adjacency: [n, n] soft adjacency matrix from LearnableAdjacency
            compressor: shared LatentCompressor
            task_token_ids: [batch_size, task_seq_len]
            task_attention_mask: [batch_size, task_seq_len] or None

        Returns:
            dict with:
                - final_hidden: [B, seq_len, D] last agent's hidden states
                - final_logits: not included here (computed by base_model separately)
                - all_hidden: list of S_i for each agent
                - all_prefixes: list of P_i for each agent (compressed)
        """
        n = len(agents)
        B = task_token_ids.shape[0]

        all_hidden = []      # full S_i for each agent (for compressor)
        all_prefixes = []    # compressed P_i for each agent
        final_logits = None  # text-only logits from terminal agent

        for j in range(n):
            # ── Step 1: Aggregate upstream prefixes ──
            upstream_prefix = self.aggregator.aggregate(
                agent_index=j,
                adjacency=adjacency,
                all_prefixes=all_prefixes,
            )

            # ── Step 2: Agent reasoning ──
            agent_output = agents[j].reason(
                task_token_ids=task_token_ids,
                task_attention_mask=task_attention_mask,
                upstream_prefix=upstream_prefix,
            )
            S_j = agent_output["full_hidden"]  # [B, full_seq, D]
            all_hidden.append(S_j)

            # ── Step 3: Compress for downstream agents ──
            if j < n - 1:
                P_j = compressor(S_j)  # [B, Lp, D]
                all_prefixes.append(P_j)
            else:
                all_prefixes.append(None)
                # Terminal agent: capture text-only logits for loss
                final_logits = agent_output["logits"]  # [B, text_len, V]

        return {
            "final_hidden": all_hidden[-1],
            "final_logits": final_logits,
            "all_hidden": all_hidden,
            "all_prefixes": all_prefixes,
        }