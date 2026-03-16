# src/pipeline/multi_agent_system.py
"""
MultiAgentSystem: the top-level module that orchestrates everything.

This is the single nn.Module you pass to the optimizer.
It owns all trainable parameters:
  - LatentCompressor (shared)
  - LearnableAdjacency

And holds references to non-trainable components:
  - BaseModelWrapper (frozen)
  - Agents (wrappers, no own params)
  - DAGExecutor (stateless orchestrator)

Usage:
    system = MultiAgentSystem(config)
    output = system(task_token_ids, task_attention_mask)
    loss = output["loss"]
    loss.backward()  # gradients flow to compressor + adjacency only
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from ..models.base_model import BaseModelWrapper
from ..models.compressor import LatentCompressor
from ..models.agent import Agent
from ..graph.adjacency import LearnableAdjacency
from ..graph.dag_executor import DAGExecutor
from ..communication.aggregator import MessageAggregator
from ..losses.task_loss import TaskLoss
from ..losses.graph_loss import GraphLoss


class MultiAgentSystem(nn.Module):
    """Top-level multi-agent latent communication system."""

    def __init__(self, config: dict):
        """
        Args:
            config: full experiment config dict (loaded from YAML)
        """
        super().__init__()
        self.config = config

        # ── Load graph config ──
        graph_config_path = config["graph"]["config"]
        with open(graph_config_path, "r") as f:
            graph_config = json.load(f)

        self.agent_roles = graph_config["agents"]
        self.n_agents = len(self.agent_roles)
        self.terminal_agent_index = graph_config["terminal_agent_index"]
        prior = torch.tensor(graph_config["adjacency_prior"], dtype=torch.float32)

        # ── Load base model (frozen, shared) ──
        self.base_model = BaseModelWrapper(
            model_name=config["model"]["name"],
            cache_dir=config["model"].get("cache_dir", "./weights"),
        )
        hidden_dim = self.base_model.hidden_dim

        # ── Create agents ──
        roles_dir = Path(config["graph"]["roles_dir"])
        self.agents = []
        for i, role_name in enumerate(self.agent_roles):
            role_path = roles_dir / f"{role_name}.json"
            with open(role_path, "r") as f:
                role_config = json.load(f)
            # Override reasoning steps from experiment config if provided
            if "reasoning" in config and "steps_per_agent" in config["reasoning"]:
                role_config["reasoning_steps"] = config["reasoning"]["steps_per_agent"]
            if "compress_last_k" in config["reasoning"]:
                role_config["compress_last_k"] = config["reasoning"]["compress_last_k"]
            agent = Agent(
                agent_id=i,
                role_config=role_config,
                base_model=self.base_model,
                max_seq_len=config["training"].get("max_seq_len", 512),
            )
            self.agents.append(agent)

        # ── Trainable: Compressor ──
        compressor_cfg = config.get("compressor", {})
        self.compressor = LatentCompressor(
            hidden_dim=hidden_dim,
            num_queries=compressor_cfg.get("num_queries", 16),
            num_heads=compressor_cfg.get("num_heads", 8),
            dropout=compressor_cfg.get("dropout", 0.1),
        )

        # ── Trainable: Adjacency ──
        self.adjacency = LearnableAdjacency(prior=prior)

        # ── Executor (stateless) ──
        self.executor = DAGExecutor(aggregator=MessageAggregator())

        # ── Losses ──
        self.task_loss_fn = TaskLoss()
        training_cfg = config.get("training", {})
        self.graph_loss_fn = GraphLoss(
            lambda_add=training_cfg.get("lambda_add", 0.1),
            lambda_drop=training_cfg.get("lambda_drop", 0.5),
            lambda_sparse=training_cfg.get("lambda_sparse", 0.1),
        )

    def forward(
        self,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
    ) -> dict:
        """Full forward pass: execute DAG + compute losses.

        Args:
            task_token_ids: [batch_size, task_seq_len]
            task_attention_mask: [batch_size, task_seq_len]
            labels: [batch_size, label_seq_len] ground truth tokens for loss.
                    If None, only returns hidden states (inference mode).

        Returns:
            dict with:
                - loss: total loss (if labels provided)
                - task_loss: task-specific loss
                - graph_loss: graph regularization loss
                - graph_loss_add: new-edge penalty
                - graph_loss_drop: dropped-edge penalty
                - final_hidden: [B, seq_len, D] terminal agent's hidden states
                - adjacency: [n, n] current soft adjacency matrix
        """
        # Get current adjacency
        A = self.adjacency.get_adjacency()

        # Execute DAG
        dag_output = self.executor.execute(
            agents=self.agents,
            adjacency=A,
            compressor=self.compressor,
            task_token_ids=task_token_ids,
            task_attention_mask=task_attention_mask,
        )

        final_logits = dag_output["final_logits"]  # [B, task_len, V]

        result = {
            "final_logits": final_logits,
            "adjacency": A,
        }

        if labels is not None:
            task_loss = self.task_loss_fn(final_logits, labels)
            graph_loss_dict = self.graph_loss_fn(A, self.adjacency.prior)
            total_loss = task_loss + graph_loss_dict["loss"]
            result.update({
                "loss": total_loss,
                "task_loss": task_loss,
                "graph_loss": graph_loss_dict["loss"],
                "graph_loss_add": graph_loss_dict["loss_add"],
                "graph_loss_drop": graph_loss_dict["loss_drop"],
                "graph_loss_sparse": graph_loss_dict["loss_sparse"],
            })

        return result

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return only the trainable parameters (for optimizer)."""
        params = []
        params.extend(self.compressor.parameters())
        params.extend(self.adjacency.parameters())
        return params

    def log_adjacency(self) -> str:
        """Pretty-print the current adjacency matrix for logging."""
        A = self.adjacency.get_adjacency().detach().cpu()
        lines = [f"Adjacency ({self.n_agents}x{self.n_agents}):"]
        header = "     " + "  ".join(f"{r[:4]:>5}" for r in self.agent_roles)
        lines.append(header)
        for i, role in enumerate(self.agent_roles):
            row = "  ".join(f"{A[i,j]:.3f}" for j in range(self.n_agents))
            lines.append(f"{role[:4]:>4}  {row}")
        return "\n".join(lines)
