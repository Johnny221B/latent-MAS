# src/communication/aggregator.py
"""
MessageAggregator: aggregates latent prefixes from upstream agents.

Supports two modes:
  - "weighted_sum" (default): normalized weighted sum using adjacency weights.
      z_j = Σ_{i < j} [ A[i,j] / Σ_k A[k,j] ] * P_{i->j}
  - "concat": concatenate incoming prefixes along sequence dimension.
      z_j = cat([P_i for i where A[i,j] > threshold], dim=1)

Gradients flow into LearnableAdjacency parameters through the weights
(weighted_sum mode only).
"""

import torch


class MessageAggregator:
    """Aggregates incoming latent messages for a downstream agent."""

    def __init__(self, mode: str = "weighted_sum", concat_threshold: float = 0.01):
        """
        Args:
            mode: "weighted_sum" or "concat"
            concat_threshold: minimum adjacency weight to include a prefix in concat mode
        """
        if mode not in ("weighted_sum", "concat"):
            raise ValueError(f"Unknown aggregation mode: {mode!r}")
        self.mode = mode
        self.concat_threshold = concat_threshold

    def aggregate(
        self,
        agent_index: int,
        adjacency: torch.Tensor,
        all_prefixes: list[torch.Tensor | None],
    ) -> torch.Tensor | None:
        if self.mode == "concat":
            return self._aggregate_concat(agent_index, adjacency, all_prefixes)
        return self._aggregate_weighted_sum(agent_index, adjacency, all_prefixes)

    def _aggregate_weighted_sum(
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

    def _aggregate_concat(
        self,
        agent_index: int,
        adjacency: torch.Tensor,
        all_prefixes: list[torch.Tensor | None],
    ) -> torch.Tensor | None:
        """Concatenate incoming prefixes along the sequence dimension.

        Each prefix is scaled by its adjacency weight before concatenation,
        so gradients from task_loss can flow back to adjacency logits.

        Output shape: [B, num_incoming * Lp, D]
        """
        j = agent_index
        incoming_prefixes = []

        for i, prefix in enumerate(all_prefixes):
            if i == j:
                continue
            if prefix is None:
                continue
            weight = adjacency[i, j]
            if float(weight.detach().item()) < self.concat_threshold:
                continue
            # Scale prefix by adjacency weight to preserve gradient flow
            incoming_prefixes.append(weight * prefix)

        if not incoming_prefixes:
            return None

        # Concatenate along sequence dimension (dim=1)
        # Each prefix is [B, Lp, D], result is [B, num_incoming * Lp, D]
        return torch.cat(incoming_prefixes, dim=1)
