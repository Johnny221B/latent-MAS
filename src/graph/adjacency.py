# src/graph/adjacency.py
"""
LearnableAdjacency: parameterized adjacency matrix for the agent communication graph.

Design:
  - Stores raw logits W ∈ R^{n×n} as nn.Parameter
  - Applies sigmoid: A = σ(W), so edge weights ∈ (0, 1)
  - Initialized from prior A(0): prior edges get large positive logits,
    non-prior edges get large negative logits
  - DAG constraint enforced by masking (upper triangular for topological order)

The adjacency matrix is the SECOND trainable component (alongside the compressor).
"""

import torch
import torch.nn as nn


class LearnableAdjacency(nn.Module):
    """Learnable soft adjacency matrix for the agent DAG."""

    def __init__(
        self,
        prior: torch.Tensor,
        init_scale: float = 5.0,
    ):
        """
        Args:
            prior: [n, n] binary tensor, the classical role prior A(0).
                   prior[i][j] = 1 means edge i->j exists in the default config.
            init_scale: magnitude for initializing logits.
                   Prior edges initialized to +init_scale (sigmoid ≈ 1),
                   non-prior edges initialized to -init_scale (sigmoid ≈ 0).
        """
        super().__init__()
        n = prior.shape[0]
        self.n = n

        # Store the prior for loss computation
        self.register_buffer("prior", prior.float())

        # Initialize logits: prior=1 -> +scale, prior=0 -> -scale
        init_logits = torch.where(
            prior > 0.5,
            torch.full_like(prior, init_scale, dtype=torch.float32),
            torch.full_like(prior, -init_scale, dtype=torch.float32),
        )

        # Zero out diagonal (no self-loops)
        init_logits.fill_diagonal_(float("-inf"))

        # Zero out lower triangle to enforce DAG (assuming topological order = agent index)
        # This means only edges i->j where i < j are allowed
        lower_mask = torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=0)
        init_logits[lower_mask] = float("-inf")

        self.logits = nn.Parameter(init_logits)

    def get_adjacency(self) -> torch.Tensor:
        """Return the soft adjacency matrix A = σ(W).

        Returns:
            A: [n, n] tensor with values in (0, 1).
                A[i][j] = connection strength from agent i to agent j.
                Diagonal and lower triangle are always 0.
        """
        return torch.sigmoid(self.logits)

    def get_hard_adjacency(self, threshold: float = 0.5) -> torch.Tensor:
        """Return a binarized adjacency matrix (for inference/visualization).

        Args:
            threshold: edges with weight >= threshold are kept.

        Returns:
            A_hard: [n, n] binary tensor.
        """
        return (self.get_adjacency() >= threshold).float()

    def forward(self) -> torch.Tensor:
        """Alias for get_adjacency()."""
        return self.get_adjacency()
