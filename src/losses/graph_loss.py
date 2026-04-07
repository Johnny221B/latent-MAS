# src/losses/graph_loss.py
"""
GraphLoss: regularization loss for the learnable adjacency matrix.

Three components:
  1. L_bce (structure-preserving): weighted BCE against prior, penalizes
     deviation from the initial graph topology.
  2. L_sparse (edge count): soft count of active edges — encourages
     turning off unused edges to reduce communication cost.
  3. L_concentrate (entropy): per-receiver entropy of normalized incoming
     edge weights — encourages each agent to attend to fewer upstream
     agents rather than spreading attention uniformly.

L_graph = λ_struct * L_bce + λ_sparse * L_sparse + λ_conc * L_concentrate
"""

import torch
import torch.nn as nn


class GraphLoss(nn.Module):
    """Graph regularization: BCE + edge-count sparsity + concentration."""

    def __init__(
        self,
        lambda_struct: float = 0.0,
        lambda_sparse: float = 0.01,
        lambda_concentrate: float = 0.1,
        w_add: float = 1.0,
        w_drop: float = 3.0,
        # Backward-compat: old params are accepted but ignored.
        lambda_add: float | None = None,
        lambda_drop: float | None = None,
    ):
        super().__init__()
        self.lambda_struct = lambda_struct
        self.lambda_sparse = lambda_sparse
        self.lambda_concentrate = lambda_concentrate
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
            dict with loss, loss_bce, loss_sparse, loss_concentrate
        """
        n = adjacency.shape[0]
        if valid_mask is None:
            valid_mask = torch.triu(
                torch.ones(n, n, device=adjacency.device, dtype=torch.bool),
                diagonal=1,
            )
        else:
            valid_mask = valid_mask.to(device=adjacency.device, dtype=torch.bool)

        eps = 1e-7

        # ── 1. Weighted BCE (structure-preserving) ──
        A_flat = adjacency[valid_mask].clamp(eps, 1.0 - eps)
        A0_flat = prior[valid_mask].float()
        per_edge_weight = torch.where(A0_flat > 0.5, self.w_drop, self.w_add)
        bce = -(A0_flat * torch.log(A_flat) + (1.0 - A0_flat) * torch.log(1.0 - A_flat))
        loss_bce = (per_edge_weight * bce).mean()

        # ── 2. Sparse (soft edge count) ──
        # Each edge contributes its weight (0~1) to the count.
        # Edges near 0 contribute ~0, edges near 0.5 contribute ~0.5.
        loss_sparse = adjacency[valid_mask].sum()

        # ── 3. Concentration (per-receiver entropy) ──
        # For each receiver j, compute entropy of its normalized incoming weights.
        # Low entropy = attention concentrated on few edges (good).
        # High entropy = attention spread uniformly (penalized).
        loss_concentrate = torch.tensor(0.0, device=adjacency.device)
        num_receivers = 0
        for j in range(n):
            incoming = valid_mask[:, j]
            if not incoming.any():
                continue
            weights = adjacency[incoming, j]
            weight_sum = weights.sum()
            if weight_sum < eps:
                continue
            p = weights / weight_sum
            entropy = -(p * torch.log(p + eps)).sum()
            loss_concentrate = loss_concentrate + entropy
            num_receivers += 1
        if num_receivers > 0:
            loss_concentrate = loss_concentrate / num_receivers

        # ── Total ──
        loss = (
            self.lambda_struct * loss_bce
            + self.lambda_sparse * loss_sparse
            + self.lambda_concentrate * loss_concentrate
        )

        return {
            "loss": loss,
            "loss_bce": loss_bce,
            "loss_sparse": loss_sparse,
            "loss_concentrate": loss_concentrate,
        }
