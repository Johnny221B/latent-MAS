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


def build_topological_edge_mask(
    execution_order: list[int],
    terminal_agent_index: int,
) -> torch.Tensor:
    """Build the boolean mask of edges allowed by the configured topological order."""
    n = len(execution_order)
    if n == 0:
        raise ValueError("execution_order must not be empty")
    if len(set(execution_order)) != n or sorted(execution_order) != list(range(n)):
        raise ValueError("execution_order must be a permutation of all agent indices")
    if terminal_agent_index not in execution_order:
        raise ValueError("terminal_agent_index must appear in execution_order")
    if execution_order[-1] != terminal_agent_index:
        raise ValueError("terminal agent must be the last node in execution_order")

    order_pos = {agent_idx: pos for pos, agent_idx in enumerate(execution_order)}
    mask = torch.zeros(n, n, dtype=torch.bool)
    for sender in range(n):
        for receiver in range(n):
            if sender == receiver:
                continue
            if order_pos[sender] < order_pos[receiver]:
                mask[sender, receiver] = True
    return mask


def validate_graph_topology(
    prior: torch.Tensor,
    execution_order: list[int],
    terminal_agent_index: int,
) -> torch.Tensor:
    """Validate the configured graph topology and return its allowed edge mask."""
    if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
        raise ValueError("adjacency_prior must be a square matrix")
    allowed_edges_mask = build_topological_edge_mask(
        execution_order=execution_order,
        terminal_agent_index=terminal_agent_index,
    )
    if prior.shape != allowed_edges_mask.shape:
        raise ValueError("adjacency_prior shape does not match execution_order length")

    if torch.any(prior[terminal_agent_index] > 0.5):
        raise ValueError("terminal agent cannot have outgoing edges in adjacency_prior")

    invalid_edges = (prior > 0.5) & ~allowed_edges_mask
    if torch.any(invalid_edges):
        raise ValueError("adjacency_prior contains edges that violate execution_order")

    return allowed_edges_mask


class LearnableAdjacency(nn.Module):
    """Learnable soft adjacency matrix for the agent DAG."""

    def __init__(
        self,
        prior: torch.Tensor,
        allowed_edges_mask: torch.Tensor | None = None,
        init_scale: float = 6.0,
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

        if allowed_edges_mask is None:
            allowed_edges_mask = torch.triu(
                torch.ones(n, n, dtype=torch.bool),
                diagonal=1,
            )
        elif allowed_edges_mask.shape != prior.shape:
            raise ValueError("allowed_edges_mask must have the same shape as prior")
        self.register_buffer("allowed_edges_mask", allowed_edges_mask.to(dtype=torch.bool))

        # Initialize logits: prior=1 -> +scale, prior=0 -> -scale
        init_logits = torch.where(
            prior > 0.5,
            torch.full_like(prior, init_scale, dtype=torch.float32),
            torch.full_like(prior, -init_scale, dtype=torch.float32),
        )

        # Mask out any illegal edges, including diagonal/self-loops.
        init_logits[~self.allowed_edges_mask] = float("-inf")

        self.logits = nn.Parameter(init_logits)

    def get_adjacency(self) -> torch.Tensor:
        """Return the soft adjacency matrix A = σ(W).

        Returns:
            A: [n, n] tensor with values in (0, 1).
                A[i][j] = connection strength from agent i to agent j.
                Illegal edges in allowed_edges_mask are always 0.
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
