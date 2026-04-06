# src/communication/aggregator.py
"""
MessageAggregator: aggregates latent prefixes from upstream agents.

Current implementation: normalized weighted sum using adjacency weights.
    z_j = Σ_{i < j} [ A[i,j] / Σ_k A[k,j] ] * P_{i->j}

The adjacency weights (sigmoid outputs) are normalized per receiver so that
the aggregated prefix scale is independent of the number of incoming edges.
Gradients flow into the LearnableAdjacency parameters through the weights.
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
        incoming_weights = []
        incoming_prefixes = []

        for i, prefix in enumerate(all_prefixes):
            if i == j:
                continue
            if prefix is None:
                continue

            weight = adjacency[i, j]
            incoming_weights.append(weight)
            incoming_prefixes.append(prefix)

        if not incoming_prefixes:
            return None

        # Normalize: divide each weight by the sum of all incoming weights
        weights = torch.stack(incoming_weights)  # [num_incoming]
        weight_sum = weights.sum()
        # Avoid division by zero (all weights near 0 means no real incoming signal)
        normalized = weights / (weight_sum + 1e-8)

        z_j = torch.stack(
            [w * p for w, p in zip(normalized, incoming_prefixes)],
            dim=0,
        ).sum(dim=0)
        return z_j
