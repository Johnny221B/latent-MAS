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
from ..models.compressor import LatentCompressor, PrefixProjector, HiddenProjection
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

        # ── Load base models ──
        # Support heterogeneous models: config.model can specify per-agent models
        # via "agent_models" dict mapping agent index/role to a model key,
        # and "models" dict mapping model keys to model paths.
        #
        # Homogeneous (default):
        #   model:
        #     name: "path/to/model"
        #
        # Heterogeneous:
        #   model:
        #     name: "path/to/primary_model"        # default / terminal model
        #     models:
        #       primary: "path/to/large_model"
        #       secondary: "path/to/small_model"
        #     agent_models: [primary, secondary, secondary, secondary, primary, primary]
        #
        model_cfg = config["model"]
        agent_model_assignments = model_cfg.get("agent_models", None)

        if agent_model_assignments and "models" in model_cfg:
            # ── Heterogeneous: load each unique model once ──
            models_dict = model_cfg["models"]
            self._base_models = nn.ModuleDict()
            for model_key, model_path in models_dict.items():
                self._base_models[model_key] = BaseModelWrapper(
                    model_name=model_path,
                    cache_dir=model_cfg.get("cache_dir"),
                    dtype=model_cfg.get("dtype"),
                )
            # Map each agent index to its model key
            self._agent_model_keys = agent_model_assignments
            # Primary model = the one used by the terminal agent
            terminal_model_key = agent_model_assignments[self.terminal_agent_index]
            self.base_model = self._base_models[terminal_model_key]
        else:
            # ── Homogeneous: single shared model (backward compatible) ──
            self.base_model = BaseModelWrapper(
                model_name=model_cfg["name"],
                cache_dir=model_cfg.get("cache_dir"),
                dtype=model_cfg.get("dtype"),
            )
            self._base_models = None
            self._agent_model_keys = None

        if config.get("training", {}).get("train_strategy") == "full_finetune":
            self.base_model.set_trainable(True)

        # Enable gradient checkpointing for E2E gradient flow through frozen model
        if config.get("training", {}).get("e2e_gradient", False):
            if self._base_models is not None:
                for model_wrapper in self._base_models.values():
                    model_wrapper.enable_gradient_checkpointing()
            else:
                self.base_model.enable_gradient_checkpointing()

        # Canonical hidden_dim = primary (terminal) model's dimension
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
            if "reasoning" in config and "compress_last_k" in config["reasoning"]:
                role_config["compress_last_k"] = config["reasoning"]["compress_last_k"]
            role_config["enable_thinking"] = config.get("model", {}).get("enable_thinking", False)
            global_sp = config.get("training", {}).get("global_system_prompt", None)
            if global_sp:
                sp_path = Path(global_sp)
                if sp_path.is_file():
                    if sp_path.suffix == ".json":
                        import json as _json
                        role_config["global_system_prompt"] = _json.loads(sp_path.read_text())["system_prompt"]
                    else:
                        role_config["global_system_prompt"] = sp_path.read_text().strip()
                else:
                    role_config["global_system_prompt"] = global_sp
            # Assign per-agent model
            if self._base_models is not None:
                agent_base_model = self._base_models[self._agent_model_keys[i]]
            else:
                agent_base_model = self.base_model
            agent = Agent(
                agent_id=i,
                role_config=role_config,
                base_model=agent_base_model,
                max_seq_len=config["training"].get("max_seq_len", 512),
            )
            self.agents.append(agent)

        # ── Trainable: Hidden projections (for heterogeneous models) ──
        # If an agent's model has a different hidden_dim than the canonical dim,
        # create a projection layer to align it before compression.
        self.hidden_projections = nn.ModuleDict()
        if self._base_models is not None:
            for model_key, model_wrapper in self._base_models.items():
                if model_wrapper.hidden_dim != hidden_dim:
                    self.hidden_projections[model_key] = HiddenProjection(
                        source_dim=model_wrapper.hidden_dim,
                        target_dim=hidden_dim,
                    )
        # Build per-agent projection index for DAGExecutor
        # None means no projection needed (same dim as canonical)
        self._agent_projection_keys = []
        for i in range(self.n_agents):
            if self._agent_model_keys is not None:
                key = self._agent_model_keys[i]
                if key in self.hidden_projections:
                    self._agent_projection_keys.append(key)
                else:
                    self._agent_projection_keys.append(None)
            else:
                self._agent_projection_keys.append(None)

        # ── Trainable: Compressor (operates in canonical hidden_dim) ──
        with torch.no_grad():
            embed_weight = self.base_model.model.get_input_embeddings().weight
            target_norm = embed_weight.norm(dim=1).mean().item()

        compressor_cfg = config.get("compressor", {})
        per_agent_compressor = compressor_cfg.get("per_agent", False)

        def _make_compressor():
            return LatentCompressor(
                hidden_dim=hidden_dim,
                num_queries=compressor_cfg.get("num_queries", 16),
                num_heads=compressor_cfg.get("num_heads", 8),
                dropout=compressor_cfg.get("dropout", 0.1),
                num_layers=compressor_cfg.get("num_layers", 1),
                target_norm=target_norm,
            )

        # ── Communication mode (latent vs pure_prefix ablation) ──
        self.communication_mode = config.get("communication", {}).get("mode", "latent")

        if self.communication_mode == "pure_prefix":
            # No compressor needed: prefix comes from per-agent learnable embeddings
            self.compressor = None
            self.compressors = None
        elif per_agent_compressor:
            # Each non-terminal agent gets its own compressor
            self.compressors = nn.ModuleList([_make_compressor() for _ in range(self.n_agents)])
            self.compressor = None  # not used
        else:
            self.compressor = _make_compressor()
            self.compressors = None

        # ── Trainable: PrefixProjectors (per unique model architecture) ──
        # Each model type needs its own PrefixProjector because KV cache
        # structure (num_layers, num_kv_heads, head_dim) differs per model.
        if self._base_models is not None:
            self.prefix_projectors = nn.ModuleDict()
            for model_key, model_wrapper in self._base_models.items():
                hf_cfg = model_wrapper.model.config
                self.prefix_projectors[model_key] = PrefixProjector(
                    num_layers=hf_cfg.num_hidden_layers,
                    hidden_dim=hidden_dim,  # input is always canonical dim
                    num_kv_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
                    head_dim=_get_kv_head_dim(hf_cfg),
                    cache_config=hf_cfg,
                )
            self.prefix_projector = None  # use per-model projectors instead
        else:
            hf_config = self.base_model.model.config
            per_agent_pp = compressor_cfg.get("per_agent_pp", False)
            if per_agent_pp:
                # One PP per agent breaks gradient symmetry when multiple agents
                # receive from the same upstream (e.g. planner→analyst/critic/verifier).
                # Keys are str(agent_index); dag_executor looks them up via agent_model_keys.
                self.prefix_projectors = nn.ModuleDict()
                for i in range(self.n_agents):
                    self.prefix_projectors[str(i)] = PrefixProjector(
                        num_layers=hf_config.num_hidden_layers,
                        hidden_dim=hf_config.hidden_size,
                        num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
                        head_dim=_get_kv_head_dim(hf_config),
                        cache_config=hf_config,
                    )
                self.prefix_projector = None
                self._agent_model_keys = [str(i) for i in range(self.n_agents)]
            else:
                self.prefix_projector = PrefixProjector(
                    num_layers=hf_config.num_hidden_layers,
                    hidden_dim=hf_config.hidden_size,
                    num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
                    head_dim=_get_kv_head_dim(hf_config),
                    cache_config=hf_config,
                )
                self.prefix_projectors = None

        # ── Trainable: per-agent learnable prefix embeddings (pure_prefix mode only) ──
        if self.communication_mode == "pure_prefix":
            Lp = compressor_cfg.get("num_queries", 16)
            self.learnable_prefix_embeddings = nn.ParameterList([
                nn.Parameter(torch.randn(1, Lp, hidden_dim) * 0.02)
                for _ in range(self.n_agents)
            ])
        else:
            self.learnable_prefix_embeddings = None

        # ── Trainable: Adjacency ──
        graph_init_scale = float(config.get("graph", {}).get("init_scale", 6.0))
        graph_fixed_structure = bool(config.get("graph", {}).get("fixed_structure", False))
        graph_noise_scale = float(config.get("graph", {}).get("noise_scale", 0.0))
        self.adjacency = LearnableAdjacency(
            prior=prior,
            allowed_edges_mask=allowed_edges_mask,
            init_scale=graph_init_scale,
            fixed_structure=graph_fixed_structure,
            noise_scale=graph_noise_scale,
        )
        self.freeze_topology = bool(config.get("graph", {}).get("freeze_topology", False))
        if self.freeze_topology:
            self.adjacency.logits.requires_grad_(False)

        # ── Executor (stateless) ──
        aggregation_mode = config.get("graph", {}).get("aggregation_mode", "weighted_sum")
        self.executor = DAGExecutor(aggregator=MessageAggregator(mode=aggregation_mode))

        # ── Cast trainable components to model dtype (bf16) ──
        model_dtype = BaseModelWrapper._resolve_dtype(model_cfg.get("dtype"))
        if model_dtype != torch.float32:
            if self.compressors is not None:
                self.compressors.to(dtype=model_dtype)
            elif self.compressor is not None:
                self.compressor.to(dtype=model_dtype)
            if self.prefix_projectors is not None:
                self.prefix_projectors.to(dtype=model_dtype)
            elif self.prefix_projector is not None:
                self.prefix_projector.to(dtype=model_dtype)
            if self.hidden_projections:
                self.hidden_projections.to(dtype=model_dtype)
            self.adjacency.to(dtype=model_dtype)
            if self.learnable_prefix_embeddings is not None:
                self.learnable_prefix_embeddings.to(dtype=model_dtype)

        # ── Losses ──
        self.task_loss_fn = TaskLoss()  # "ce" or "reward"
        training_cfg = config.get("training", {})
        self.graph_loss_fn = GraphLoss(
            lambda_struct=training_cfg.get("lambda_struct", 0.0),
            lambda_sparse=training_cfg.get("lambda_sparse", 0.01),
            lambda_concentrate=training_cfg.get("lambda_concentrate", 0.1),
            w_add=training_cfg.get("w_add", 1.0),
            w_drop=training_cfg.get("w_drop", 3.0),
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
            compressors=self.compressors,
            prefix_projector=self.prefix_projector,
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
            # Heterogeneous model support
            hidden_projections=self.hidden_projections if self.hidden_projections else None,
            agent_projection_keys=self._agent_projection_keys,
            prefix_projectors=self.prefix_projectors,
            agent_model_keys=self._agent_model_keys,
            e2e_gradient=bool(self.config.get("training", {}).get("e2e_gradient", False)),
            communication_mode=self.communication_mode,
            learnable_prefix_embeddings=list(self.learnable_prefix_embeddings) if self.learnable_prefix_embeddings is not None else None,
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
            if self.freeze_topology:
                graph_loss_dict = {
                    "loss": torch.tensor(0.0, device=task_loss.device),
                    "loss_bce": torch.tensor(0.0, device=task_loss.device),
                    "loss_sparse": torch.tensor(0.0, device=task_loss.device),
                    "loss_concentrate": torch.tensor(0.0, device=task_loss.device),
                }
            else:
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
                "graph_loss_bce": graph_loss_dict["loss_bce"],
                "graph_loss_sparse": graph_loss_dict["loss_sparse"],
                "graph_loss_concentrate": graph_loss_dict["loss_concentrate"],
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
        if self.compressors is not None:
            params.extend(self.compressors.parameters())
        elif self.compressor is not None:
            params.extend(self.compressor.parameters())
        if self.learnable_prefix_embeddings is not None:
            params.extend(self.learnable_prefix_embeddings.parameters())
        if not self.freeze_topology:
            params.extend(self.adjacency.parameters())
        # Hidden projections (heterogeneous model alignment)
        if self.hidden_projections:
            params.extend(self.hidden_projections.parameters())
        # Prefix projectors
        if self.prefix_projectors is not None:
            params.extend(self.prefix_projectors.parameters())
        elif self.prefix_projector is not None:
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
