# src/losses/graph_loss.py
"""
GraphLoss: regularization loss for the learnable adjacency matrix.

Implements the asymmetric penalty from the paper:
  L_graph = λ_add * L_add + λ_drop * L_drop

Where:
  L_add  = Σ_{(i,j): A0[i,j]=0} A[i,j]       — penalize opening new edges
  L_drop = Σ_{(i,j): A0[i,j]=1} (1 - A[i,j])  — penalize closing prior edges

The asymmetry (λ_add ≠ λ_drop) encodes the inductive bias:
  - High λ_drop: "trust the classical pipeline, don't easily remove edges"
  - Low λ_add: "allow exploration of new connections if helpful"
"""

import torch
import torch.nn as nn


class GraphLoss(nn.Module):
    """Asymmetric graph regularization loss."""

    def __init__(self, lambda_add: float = 0.1, lambda_drop: float = 0.5, lambda_sparse: float = 0.1):
        """
        Args:
            lambda_add: weight for penalizing new edges (not in prior)
            lambda_drop: weight for penalizing dropped edges (in prior)
        """
        super().__init__()
        self.lambda_add = lambda_add
        self.lambda_drop = lambda_drop
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        adjacency: torch.Tensor,
        prior: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute graph regularization loss.

        Args:
            adjacency: [n, n] current soft adjacency matrix (sigmoid output)
            prior: [n, n] binary prior matrix A(0)

        Returns:
            dict with:
                - loss: total graph loss (scalar)
                - loss_add: penalty for new edges
                - loss_drop: penalty for dropped edges
        """
        # Mask for prior edges and non-prior edges
        prior_mask = (prior > 0.5)        # edges that exist in classical config
        non_prior_mask = ~prior_mask       # edges that don't exist in classical config

        # Also exclude diagonal and lower triangle (these are always -inf in logits)
        n = adjacency.shape[0]
        valid_mask = torch.triu(torch.ones(n, n, device=adjacency.device, dtype=torch.bool), diagonal=1)

        # L_add: sum of adjacency weights on non-prior, valid edges
        add_mask = non_prior_mask & valid_mask
        loss_add = adjacency[add_mask].sum()

        # L_drop: sum of (1 - adjacency) on prior edges
        drop_mask = prior_mask & valid_mask
        loss_drop = (1.0 - adjacency[drop_mask]).sum()

        # Total
        # L1 sparsity: penalize all non-zero edge weights in upper triangle
        loss_sparse = adjacency[valid_mask].sum()

        # Total
        loss = (
            self.lambda_add * loss_add
            + self.lambda_drop * loss_drop
            + self.lambda_sparse * loss_sparse
        )

        return {
            "loss": loss,
            "loss_add": loss_add,
            "loss_drop": loss_drop,
            "loss_sparse": loss_sparse,
        }
