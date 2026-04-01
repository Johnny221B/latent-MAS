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
from ..models.compressor import LatentCompressor, PrefixProjector
from ..models.agent import Agent
from ..graph.adjacency import (
    LearnableAdjacency,
    validate_graph_topology,
)
from ..graph.dag_executor import DAGExecutor
from ..communication.aggregator import MessageAggregator
from ..losses.task_loss import TaskLoss
from ..losses.graph_loss import GraphLoss


def _get_kv_head_dim(hf_config) -> int:
    return getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)


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
        self.execution_order = graph_config.get("execution_order", list(range(self.n_agents)))
        self.training_input_mode = config.get("training", {}).get("input_mode", "legacy_plain_with_prefix")
        prior = torch.tensor(graph_config["adjacency_prior"], dtype=torch.float32)
        allowed_edges_mask = validate_graph_topology(
            prior=prior,
            execution_order=self.execution_order,
            terminal_agent_index=self.terminal_agent_index,
        )

        # ── Load base model (frozen, shared) ──
        self.base_model = BaseModelWrapper(
            model_name=config["model"]["name"],
            cache_dir=config["model"].get("cache_dir"),
            dtype=config["model"].get("dtype"),
        )
        if config.get("training", {}).get("train_strategy") == "full_finetune":
            self.base_model.set_trainable(True)
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
        # ── Compute target norm from input embeddings ──
        with torch.no_grad():
            embed_weight = self.base_model.model.get_input_embeddings().weight
            target_norm = embed_weight.norm(dim=1).mean().item()

        # ── Trainable: Compressor ──
        compressor_cfg = config.get("compressor", {})
        self.compressor = LatentCompressor(
            hidden_dim=hidden_dim,
            num_queries=compressor_cfg.get("num_queries", 16),
            num_heads=compressor_cfg.get("num_heads", 8),
            dropout=compressor_cfg.get("dropout", 0.1),
            num_layers=compressor_cfg.get("num_layers", 1),
            target_norm=target_norm,
        )
        
        hf_config = self.base_model.model.config
        self.prefix_projector = PrefixProjector(
            num_layers=hf_config.num_hidden_layers,
            hidden_dim=hf_config.hidden_size,
            num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            head_dim=_get_kv_head_dim(hf_config),
            cache_config=hf_config,
        )

        # ── Trainable: Adjacency ──
        self.adjacency = LearnableAdjacency(
            prior=prior,
            allowed_edges_mask=allowed_edges_mask,
            init_scale=6.0,
        )

        # ── Executor (stateless) ──
        self.executor = DAGExecutor(aggregator=MessageAggregator())

        # ── Losses ──
        self.task_loss_fn = TaskLoss()  # "ce" or "reward"
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
        answer_ids: torch.LongTensor | None = None,
        answer_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        inference_mode: str = "legacy_plain_with_prefix",
        use_terminal_prefix: bool = True,
        do_sample: bool = True,
        collect_agent_logs: bool = False,
    ) -> dict:
        A = self.adjacency.get_adjacency()
        is_training = answer_ids is not None

        dag_output = self.executor.execute(
            agents=self.agents,
            adjacency=A,
            compressor=self.compressor,
            prefix_projector=self.prefix_projector, # prefix
            task_token_ids=task_token_ids,
            task_attention_mask=task_attention_mask,
            training=is_training,
            answer_ids=answer_ids,
            answer_mask=answer_mask,
            max_new_tokens=max_new_tokens,
            inference_mode=inference_mode,
            use_terminal_prefix=use_terminal_prefix,
            do_sample=do_sample,
            collect_agent_logs=collect_agent_logs,
            execution_order=self.execution_order,
            terminal_agent_index=self.terminal_agent_index,
            training_input_mode=self.training_input_mode,
        )

        result = {"adjacency": A}

        if is_training:
            final_logits = dag_output["final_logits"]
            question_len = dag_output["question_len"]

            # Build labels: [-100, ..., -100, answer_tokens]
            #               |-- question --|--- answer ---|
            from src.data import build_labels
            labels = build_labels(
                question_len=question_len,
                answer_ids=answer_ids,
            )

            task_loss = self.task_loss_fn(final_logits, labels)
            graph_loss_dict = self.graph_loss_fn(
                A,
                self.adjacency.prior,
                valid_mask=self.adjacency.allowed_edges_mask,
            )
            total_loss = task_loss + graph_loss_dict["loss"]

            result.update({
                "loss": total_loss,
                "task_loss": task_loss,
                "final_logits": final_logits,
                "graph_loss": graph_loss_dict["loss"],
                "graph_loss_add": graph_loss_dict["loss_add"],
                "graph_loss_drop": graph_loss_dict["loss_drop"],
                "graph_loss_sparse": graph_loss_dict["loss_sparse"],
            })
        else:
            result["generated_text"] = dag_output["generated_text"]
            result["generation"] = dag_output["generation"]
            if collect_agent_logs:
                result["agent_logs"] = dag_output.get("agent_logs", [])

        return result

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return only the trainable parameters (for optimizer)."""
        params = []
        if self.config.get("training", {}).get("train_strategy") == "full_finetune":
            params.extend(self.base_model.model.parameters())
        params.extend(self.compressor.parameters())
        params.extend(self.adjacency.parameters())
        params.extend(self.prefix_projector.parameters())
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
