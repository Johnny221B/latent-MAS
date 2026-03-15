"""
MessageAggregator: aggregates latent prefixes from upstream agents.

Current implementation: weighted sum using adjacency weights.
    z_j = Σ_{i ∈ N(j)} A[i,j] * P_{i->j}

This is where adjacency weights directly multiply the prefix tensors,
allowing gradients to flow into the LearnableAdjacency parameters.

Extensibility:
  - Attention-based aggregation: replace weighted sum with learned attention.
  - Concatenation: concat all prefixes (variable length, need different compressor).
"""

import torch


class MessageAggregator:
    """Aggregates incoming latent messages for a downstream agent."""

    def aggregate(
        self,
        agent_index: int,
        adjacency: torch.Tensor,
        all_prefixes: list[torch.Tensor | None],
    ) -> torch.Tensor | None:
        """Compute the aggregated prefix for agent j from all upstream agents.

        Args:
            agent_index: index j of the receiving agent
            adjacency: [n, n] soft adjacency matrix
            all_prefixes: list of compressed prefixes P_i for agents 0..j-1
                         (None entries are skipped)

        Returns:
            z_j: [batch_size, Lp, hidden_dim] aggregated prefix, or None if no upstream.
        """
        j = agent_index
        incoming = []

        for i in range(j):  # only upstream agents (i < j due to DAG)
            if all_prefixes[i] is None:
                continue

            weight = adjacency[i, j]  # scalar, differentiable

            # Skip negligible connections for efficiency
            # (but keep gradient path alive — no hard thresholding during training)
            if weight.item() < 1e-6:
                continue

            # Weighted prefix: A[i,j] * P_i
            weighted_prefix = weight * all_prefixes[i]  # [B, Lp, D]
            incoming.append(weighted_prefix)

        if not incoming:
            return None

        # Sum all weighted prefixes
        z_j = torch.stack(incoming, dim=0).sum(dim=0)  # [B, Lp, D]
        return z_j
