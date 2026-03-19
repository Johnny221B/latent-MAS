# src/communication/aggregator.py
"""
MessageAggregator: aggregates latent prefixes from upstream agents.

Current implementation: weighted sum using adjacency weights.
    z_j = Σ_{i < j} A[i,j] * P_{i->j}

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
        j = agent_index
        incoming = []
        weight_sum = 0.0

        for i in range(j):
            if all_prefixes[i] is None:
                continue

            weight = adjacency[i, j]
            incoming.append(weight * all_prefixes[i])
            weight_sum = weight_sum + weight  # differentiable

        if not incoming:
            return None

        z_j = torch.stack(incoming, dim=0).sum(dim=0)

        # Normalize so incoming weights sum to 1
        z_j = z_j / (weight_sum + 1e-8)

        return z_j
