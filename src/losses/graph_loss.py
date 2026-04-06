# src/losses/graph_loss.py
"""
GraphLoss: regularization loss for the learnable adjacency matrix.

Implements weighted BCE + L1 sparsity:
  L_graph = λ_struct * L_bce + λ_sparse * L_sparse

Where:
  L_bce = -Σ [ w_drop · A0 · log(A+ε) + w_add · (1-A0) · log(1-A+ε) ]
  L_sparse = Σ A[i,j]   (all valid edges)

The asymmetry (w_add ≠ w_drop) encodes the inductive bias:
  - High w_drop: "trust the classical pipeline, penalize removing prior edges"
  - Low w_add: "allow exploration of new connections if helpful"
"""

import torch
import torch.nn as nn


class GraphLoss(nn.Module):
    """Weighted BCE + sparsity graph regularization loss."""

    def __init__(
        self,
        lambda_struct: float = 0.1,
        lambda_sparse: float = 0.01,
        w_add: float = 1.0,
        w_drop: float = 5.0,
        # Keep old params for backward-compat config loading; they are ignored.
        lambda_add: float | None = None,
        lambda_drop: float | None = None,
    ):
        """
        Args:
            lambda_struct: overall weight for the structure-preserving BCE loss.
            lambda_sparse: weight for L1 sparsity penalty on all valid edges.
            w_add: weight inside BCE for non-prior edges (penalizes adding).
            w_drop: weight inside BCE for prior edges (penalizes dropping).
        """
        super().__init__()
        self.lambda_struct = lambda_struct
        self.lambda_sparse = lambda_sparse
        self.w_add = w_add
        self.w_drop = w_drop

    def forward(
        self,
        adjacency: torch.Tensor,
        prior: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute graph regularization loss.

        Args:
            adjacency: [n, n] current soft adjacency matrix (sigmoid output)
            prior: [n, n] binary prior matrix A(0)
            valid_mask: [n, n] boolean mask of edges allowed by DAG topology

        Returns:
            dict with:
                - loss: total graph loss (scalar)
                - loss_bce: weighted BCE component
                - loss_sparse: L1 sparsity component
        """
        if valid_mask is None:
            n = adjacency.shape[0]
            valid_mask = torch.triu(
                torch.ones(n, n, device=adjacency.device, dtype=torch.bool),
                diagonal=1,
            )
        else:
            valid_mask = valid_mask.to(device=adjacency.device, dtype=torch.bool)

        eps = 1e-7
        A = adjacency[valid_mask].clamp(eps, 1.0 - eps)
        A0 = prior[valid_mask].float()

        # Weighted BCE: per-edge weight depends on whether it's in the prior
        per_edge_weight = torch.where(A0 > 0.5, self.w_drop, self.w_add)
        bce = -(A0 * torch.log(A) + (1.0 - A0) * torch.log(1.0 - A))
        loss_bce = (per_edge_weight * bce).mean()

        # L1 sparsity on all valid edges
        loss_sparse = adjacency[valid_mask].mean()

        loss = self.lambda_struct * loss_bce + self.lambda_sparse * loss_sparse

        return {
            "loss": loss,
            "loss_bce": loss_bce,
            "loss_sparse": loss_sparse,
        }
