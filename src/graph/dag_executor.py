# src/graph/dag_executor.py

import torch
from transformers.cache_utils import DynamicCache

from ..models.agent import Agent
from ..models.compressor import LatentCompressor, PrefixProjector, HiddenProjection
from ..communication.aggregator import MessageAggregator


def _compute_dag_levels(execution_order: list[int], adjacency: torch.Tensor, terminal_index: int) -> list[list[int]]:
    """Group agents into parallelizable levels based on the DAG structure.

    Two non-terminal agents can be in the same level if neither depends on the other
    (i.e., they only depend on agents in earlier levels).
    """
    n = len(execution_order)
    # Assign each agent a level = max(level of any upstream agent) + 1
    agent_level = {}
    for j in execution_order:
        max_upstream_level = -1
        for i in execution_order:
            if i == j:
                continue
            if float(adjacency[i, j].detach().item()) > 0.01 and i in agent_level:
                max_upstream_level = max(max_upstream_level, agent_level[i])
        agent_level[j] = max_upstream_level + 1

    # Group by level
    max_level = max(agent_level.values()) if agent_level else 0
    levels = []
    for lv in range(max_level + 1):
        group = [j for j in execution_order if agent_level[j] == lv]
        if group:
            levels.append(group)
    return levels


def _batch_kv_caches(caches: list[DynamicCache | None]) -> DynamicCache | None:
    """Batch multiple DynamicCache objects along the batch dimension."""
    valid = [c for c in caches if c is not None]
    if not valid:
        return None
    if len(valid) != len(caches):
        return None  # mixed None/non-None, can't batch

    num_layers = len(valid[0])
    batched = DynamicCache()
    for layer_idx in range(num_layers):
        keys = torch.cat([c[layer_idx][0] for c in valid], dim=0)
        values = torch.cat([c[layer_idx][1] for c in valid], dim=0)
        batched.update(keys, values, layer_idx)
    return batched


class DAGExecutor:
    """Executes the multi-agent DAG in topological order."""

    def __init__(self, aggregator: MessageAggregator | None = None):
        """
        Args:
            aggregator: MessageAggregator instance. If None, creates default.
        """
        self.aggregator = aggregator or MessageAggregator()

    def execute(
        self,
        agents: list[Agent],
        adjacency: torch.Tensor,
        compressor: LatentCompressor | None,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        compressors: list[LatentCompressor] | None = None,
        prefix_projector: PrefixProjector | None = None,
        training: bool = True,
        answer_ids: torch.LongTensor | None = None,
        answer_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        inference_mode: str = "legacy_plain_with_prefix",
        use_terminal_prefix: bool = True,
        do_sample: bool = True,
        collect_agent_logs: bool = False,
        execution_order: list[int] | None = None,
        terminal_agent_index: int | None = None,
        training_input_mode: str = "legacy_plain_with_prefix",
        # Heterogeneous model support
        hidden_projections: dict[str, HiddenProjection] | None = None,
        agent_projection_keys: list[str | None] | None = None,
        prefix_projectors: dict[str, PrefixProjector] | None = None,
        agent_model_keys: list[str] | None = None,
        e2e_gradient: bool = False,
        # Ablation: pure_prefix mode
        communication_mode: str = "latent",
        learnable_prefix_embeddings: list[torch.Tensor] | None = None,
    ) -> dict:
        """Execute all agents in topological order.

        Args:
            agents: list of Agent instances, ordered by index (= topological order)
            adjacency: [n, n] soft adjacency matrix from LearnableAdjacency
            compressor: shared LatentCompressor
            task_token_ids: [batch_size, task_seq_len]
            task_attention_mask: [batch_size, task_seq_len] or None

        Returns:
            dict with:
                - final_hidden: [B, seq_len, D] last agent's hidden states
                - final_logits: not included here (computed by base_model separately)
                - all_hidden: list of S_i for each agent
                - all_prefixes: list of P_i for each agent (compressed)
        """
        n = len(agents)
        execution_order = execution_order or list(range(n))
        terminal_index = terminal_agent_index if terminal_agent_index is not None else (n - 1)
        if len(execution_order) != n or sorted(execution_order) != list(range(n)):
            raise ValueError("execution_order must be a permutation of all agent indices")
        if execution_order[-1] != terminal_index:
            raise ValueError("terminal agent must be the last node in execution_order")

        all_prefixes = [None] * n
        all_text_outputs = [None] * n  # In pure_prefix mode: list[str] per agent (length B)
        agent_logs = []

        def collect_upstream_texts(agent_index: int) -> list[str]:
            """Collect upstream text strings for inference (single-item)."""
            texts = []
            for upstream_idx in range(n):
                message = all_text_outputs[upstream_idx]
                if message is None:
                    continue
                if float(adjacency[upstream_idx, agent_index].detach().item()) > 0.5:
                    if isinstance(message, list):
                        texts.append(message[0])
                    else:
                        texts.append(message)
            return texts

        def collect_upstream_texts_batched(agent_index: int, B: int) -> list[list[str]]:
            """Collect upstream text strings for batched training: returns [B x num_upstream_texts].

            In pure_prefix mode: collect ALL prior agents' texts (in execution order),
            so every downstream agent sees the full upstream reasoning chain.
            In other modes: filter by adjacency > 0.5.
            """
            upstream_texts_batch = [[] for _ in range(B)]
            for upstream_idx in execution_order:
                if upstream_idx == agent_index:
                    break  # only collect agents that run before this one
                message = all_text_outputs[upstream_idx]
                if message is None:
                    continue
                include = (
                    communication_mode == "pure_prefix"
                    or float(adjacency[upstream_idx, agent_index].detach().item()) > 0.5
                )
                if include:
                    if isinstance(message, list):
                        for b in range(B):
                            upstream_texts_batch[b].append(message[b] if b < len(message) else message[0])
                    else:
                        for b in range(B):
                            upstream_texts_batch[b].append(message)
            return upstream_texts_batch

        def get_prefix_projector_for(agent_index: int) -> PrefixProjector | None:
            """Resolve the correct PrefixProjector for a given agent."""
            if prefix_projectors is not None and agent_model_keys is not None:
                return prefix_projectors[agent_model_keys[agent_index]]
            return prefix_projector

        def project_hidden(agent_index: int, hidden: torch.Tensor) -> torch.Tensor:
            """Project hidden states to canonical dim if agent uses a smaller model."""
            if hidden_projections is None or agent_projection_keys is None:
                return hidden
            key = agent_projection_keys[agent_index]
            if key is None:
                return hidden
            return hidden_projections[key](hidden)

        def get_compressor_for(agent_index: int) -> LatentCompressor:
            """Resolve compressor: per-agent if available, else shared."""
            if compressors is not None:
                return compressors[agent_index]
            return compressor

        def summarize_tensor(name: str, value: torch.Tensor | None) -> dict | None:
            if value is None:
                return None
            detached = value.detach().float()
            return {
                "name": name,
                "shape": list(detached.shape),
                "norm": float(detached.norm().item()),
                "mean": float(detached.mean().item()),
                "std": float(detached.std(unbiased=False).item()),
            }

        # ── Compute parallelizable levels from DAG ──
        dag_levels = _compute_dag_levels(execution_order, adjacency, terminal_index)

        def _execute_single_nonterminal(j, upstream_prefix):
            """Execute a single non-terminal agent (original sequential path)."""
            # ── pure_prefix ablation: own prefix + text communication ──
            if communication_mode == "pure_prefix":
                B = task_token_ids.shape[0]
                own_emb = learnable_prefix_embeddings[j].expand(B, -1, -1)
                own_pp = get_prefix_projector_for(j)
                own_prefix_kv = own_pp(own_emb, target_dtype=agents[j].base_model.dtype) if own_pp is not None else None
                upstream_texts_batch = collect_upstream_texts_batched(j, B)
                text_output = agents[j].generate_answer(
                    task_token_ids=task_token_ids,
                    task_attention_mask=task_attention_mask,
                    upstream_prefix_kv=own_prefix_kv,
                    upstream_texts=upstream_texts_batch,
                    max_new_tokens=max_new_tokens,
                    inference_mode="chat_with_text",
                    use_upstream_prefix=True,
                    do_sample=do_sample,
                    return_metadata=True,
                )
                generated = text_output["generated_text"]
                # Store per-batch-item texts
                all_text_outputs[j] = generated if isinstance(generated, list) else [generated] * B
                if collect_agent_logs:
                    agent_logs.append({
                        "agent_id": agents[j].agent_id,
                        "role_name": agents[j].role_name,
                        "output_type": "pure_prefix_text",
                        "system_prompt": agents[j].system_prompt,
                        "received_upstream_texts": bool(any(upstream_texts_batch)),
                        "generated_text": generated,
                        "generation": text_output,
                    })
                return

            if not training and inference_mode == "chat_with_text":
                upstream_texts = collect_upstream_texts(j)
                text_output = agents[j].generate_answer(
                    task_token_ids=task_token_ids,
                    task_attention_mask=task_attention_mask,
                    upstream_prefix=None,
                    upstream_texts=[upstream_texts for _ in range(task_token_ids.shape[0])],
                    max_new_tokens=max_new_tokens,
                    return_metadata=True,
                    inference_mode=inference_mode,
                    use_upstream_prefix=False,
                    do_sample=do_sample,
                )
                generated_text = text_output["generated_text"]
                all_text_outputs[j] = generated_text[0] if isinstance(generated_text, list) else generated_text
                if collect_agent_logs:
                    agent_logs.append({
                        "agent_id": agents[j].agent_id,
                        "role_name": agents[j].role_name,
                        "output_type": "text_message",
                        "system_prompt": agents[j].system_prompt,
                        "received_upstream_prefix": False,
                        "upstream_prefix": None,
                        "received_upstream_texts": bool(upstream_texts),
                        "upstream_texts": upstream_texts,
                        "generated_text": generated_text,
                        "generation": text_output,
                    })
                return

            upstream_prefix_kv = None
            agent_pp = get_prefix_projector_for(j)
            if agent_pp is not None and upstream_prefix is not None:
                upstream_prefix_kv = agent_pp(upstream_prefix, target_dtype=agents[j].base_model.dtype)
            agent_output = agents[j].reason(
                task_token_ids=task_token_ids,
                task_attention_mask=task_attention_mask,
                upstream_prefix=upstream_prefix,
                upstream_prefix_kv=upstream_prefix_kv,
            )
            del upstream_prefix_kv
            S_j = agent_output["hidden_trajectory"]
            S_j = project_hidden(j, S_j)
            mask_j = agent_output["compressor_mask"]
            P_j = get_compressor_for(j)(S_j, mask=mask_j)
            all_prefixes[j] = P_j
            if collect_agent_logs:
                agent_logs.append({
                    "agent_id": agents[j].agent_id,
                    "role_name": agents[j].role_name,
                    "output_type": "latent",
                    "system_prompt": agents[j].system_prompt,
                    "received_upstream_prefix": upstream_prefix is not None,
                    "upstream_prefix": summarize_tensor("upstream_prefix", upstream_prefix),
                    "hidden_trajectory": summarize_tensor("hidden_trajectory", S_j),
                    "compressed_prefix": summarize_tensor("compressed_prefix", P_j),
                })

        def _execute_batched_nonterminals(agent_indices, upstream_prefixes):
            """Execute multiple independent non-terminal agents in a single batched forward."""
            B = task_token_ids.shape[0]
            num_agents = len(agent_indices)
            tokenizer = agents[agent_indices[0]].base_model.tokenizer

            # Decode question texts once
            question_texts = tokenizer.batch_decode(
                task_token_ids.detach().cpu(), skip_special_tokens=True,
            )

            # Build prompts for all agents (each agent has different role_prompt)
            all_prompts = []
            for j in agent_indices:
                for q in question_texts:
                    all_prompts.append(Agent.build_chat_prompt_text(
                        tokenizer=tokenizer,
                        question_text=q,
                        system_prompt=agents[j].system_prompt,
                        role_prompt=agents[j].role_prompt,
                        enable_thinking=agents[j].enable_thinking,
                    ))

            # Tokenize all prompts together (handles padding automatically)
            tokenized = tokenizer(
                all_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048, add_special_tokens=False,
            )
            batched_input_ids = tokenized["input_ids"].to(task_token_ids.device)
            batched_attention_mask = tokenized["attention_mask"].to(task_token_ids.device)

            # Build batched upstream prefix KV caches
            batched_prefix_kv = None
            batched_prefix_embeds = None
            use_kv = False

            # Check if all agents use the same prefix projector
            agent_pp = get_prefix_projector_for(agent_indices[0])
            if agent_pp is not None:
                kv_list = []
                model_dtype = agents[agent_indices[0]].base_model.dtype
                for j, up in zip(agent_indices, upstream_prefixes):
                    if up is not None:
                        kv = agent_pp(up, target_dtype=model_dtype)
                        kv_list.append(kv)
                    else:
                        kv_list.append(None)
                batched_prefix_kv = _batch_kv_caches(kv_list)
                if batched_prefix_kv is not None:
                    use_kv = True
            if not use_kv:
                # Fallback to prefix_embeds
                embeds_list = []
                for up in upstream_prefixes:
                    if up is not None:
                        embeds_list.append(up)
                if len(embeds_list) == num_agents:
                    batched_prefix_embeds = torch.cat(embeds_list, dim=0)

            # Get reasoning steps (use first agent's, they should all be the same)
            reasoning_steps = agents[agent_indices[0]].reasoning_steps
            compress_last_k = agents[agent_indices[0]].compress_last_k

            # Single batched latent_reasoning call — no no_grad so task loss
            # gradients can flow through the full DAG
            output = agents[agent_indices[0]].base_model.latent_reasoning(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_mask,
                prefix_embeds=batched_prefix_embeds if not use_kv else None,
                past_key_values=batched_prefix_kv if use_kv else None,
                num_latent_steps=reasoning_steps,
                grad_last_k=compress_last_k,
            )
            del batched_prefix_kv

            trajectory = output["hidden_trajectory"]  # [num_agents*B, m, D]
            initial_hidden_all = output.get("initial_hidden")  # [num_agents*B, input_seq_len, D] or None

            # Split back per agent
            k = min(compress_last_k, trajectory.shape[1])
            for idx, j in enumerate(agent_indices):
                agent_traj = trajectory[idx * B : (idx + 1) * B]  # [B, m, D]
                traj_to_compress = agent_traj[:, -k:, :]

                # Prepend initial encoding hidden states if available
                if initial_hidden_all is not None:
                    agent_initial = initial_hidden_all[idx * B : (idx + 1) * B]  # [B, seq_len, D]
                    agent_initial = agent_initial.to(dtype=traj_to_compress.dtype)
                    to_compress = torch.cat([agent_initial, traj_to_compress], dim=1)
                    initial_mask = torch.ones(B, agent_initial.shape[1], device=trajectory.device)
                    mask_j = torch.cat([initial_mask, torch.ones(B, k, device=trajectory.device)], dim=1)
                else:
                    to_compress = traj_to_compress
                    mask_j = torch.ones(B, k, device=trajectory.device)

                S_j = project_hidden(j, to_compress)
                P_j = get_compressor_for(j)(S_j, mask=mask_j)
                all_prefixes[j] = P_j
                if collect_agent_logs:
                    agent_logs.append({
                        "agent_id": agents[j].agent_id,
                        "role_name": agents[j].role_name,
                        "output_type": "latent_batched",
                        "system_prompt": agents[j].system_prompt,
                        "received_upstream_prefix": upstream_prefixes[idx] is not None,
                        "upstream_prefix": summarize_tensor("upstream_prefix", upstream_prefixes[idx]),
                        "hidden_trajectory": summarize_tensor("hidden_trajectory", S_j),
                        "compressed_prefix": summarize_tensor("compressed_prefix", P_j),
                    })

        for level in dag_levels:
            # Separate terminal from non-terminal agents in this level
            nonterminal_in_level = [j for j in level if j != terminal_index]
            terminal_in_level = [j for j in level if j == terminal_index]

            if len(nonterminal_in_level) > 1 and training and not e2e_gradient and communication_mode == "latent":
                # ── Batched execution for multiple independent non-terminal agents ──
                upstream_prefixes = []
                for j in nonterminal_in_level:
                    up = self.aggregator.aggregate(
                        agent_index=j, adjacency=adjacency, all_prefixes=all_prefixes,
                    )
                    upstream_prefixes.append(up)
                _execute_batched_nonterminals(nonterminal_in_level, upstream_prefixes)
            else:
                # ── Sequential execution (single agent or inference) ──
                for j in nonterminal_in_level:
                    upstream_prefix = self.aggregator.aggregate(
                        agent_index=j, adjacency=adjacency, all_prefixes=all_prefixes,
                    )
                    _execute_single_nonterminal(j, upstream_prefix)

            # ── Handle terminal agent if in this level ──
            for j in terminal_in_level:
                # ── pure_prefix ablation: own prefix + upstream text ──
                if communication_mode == "pure_prefix":
                    B = task_token_ids.shape[0]
                    own_emb = learnable_prefix_embeddings[j].expand(B, -1, -1)
                    terminal_pp = get_prefix_projector_for(j)
                    own_prefix_kv = terminal_pp(own_emb, target_dtype=agents[j].base_model.dtype) if terminal_pp is not None else None
                    upstream_texts_batch = collect_upstream_texts_batched(j, B)
                    if training:
                        terminal_output = agents[j].forward_for_loss(
                            task_token_ids=task_token_ids,
                            task_attention_mask=task_attention_mask,
                            upstream_prefix_kv=own_prefix_kv,
                            answer_ids=answer_ids,
                            answer_mask=answer_mask,
                            input_mode="chat_with_text",
                            upstream_texts=upstream_texts_batch,
                        )
                    else:
                        generation = agents[j].generate_answer(
                            task_token_ids=task_token_ids,
                            task_attention_mask=task_attention_mask,
                            upstream_prefix_kv=own_prefix_kv,
                            upstream_texts=upstream_texts_batch,
                            max_new_tokens=max_new_tokens,
                            inference_mode="chat_with_text",
                            use_upstream_prefix=True,
                            do_sample=do_sample,
                            return_metadata=True,
                        )
                        if collect_agent_logs:
                            agent_logs.append({
                                "agent_id": agents[j].agent_id,
                                "role_name": agents[j].role_name,
                                "output_type": "text",
                                "generated_text": generation["generated_text"],
                                "generation": generation,
                            })
                    continue

                upstream_prefix = self.aggregator.aggregate(
                    agent_index=j, adjacency=adjacency, all_prefixes=all_prefixes,
                )
                upstream_prefix_kv = None
                terminal_pp = get_prefix_projector_for(j)
                if terminal_pp is not None and upstream_prefix is not None:
                    upstream_prefix_kv = terminal_pp(upstream_prefix, target_dtype=agents[j].base_model.dtype)
                if training:
                    # Training: single forward pass, return logits for CE loss
                    terminal_output = agents[j].forward_for_loss(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_prefix=upstream_prefix,
                        upstream_prefix_kv=upstream_prefix_kv,
                        answer_ids=answer_ids,
                        answer_mask=answer_mask,
                        input_mode=training_input_mode,
                    )
                else:
                    # Inference: autoregressive generation, return text
                    upstream_texts = None
                    used_upstream_prefix = use_terminal_prefix
                    terminal_upstream_prefix = upstream_prefix
                    if inference_mode == "chat_with_text":
                        upstream_texts = collect_upstream_texts(j)
                        terminal_upstream_prefix = None
                        used_upstream_prefix = False
                    terminal_prefix_kv = upstream_prefix_kv if inference_mode != "chat_with_text" else None
                    terminal_prefix_len = 0
                    if terminal_prefix_kv is not None:
                        if hasattr(terminal_prefix_kv, "get_seq_length"):
                            terminal_prefix_len = int(terminal_prefix_kv.get_seq_length())
                        elif len(terminal_prefix_kv) > 0:
                            terminal_prefix_len = int(terminal_prefix_kv[0][0].shape[-2])
                    generation = agents[j].generate_answer(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_prefix=terminal_upstream_prefix,
                        upstream_prefix_kv=terminal_prefix_kv,
                        prefix_len=terminal_prefix_len,
                        upstream_texts=[upstream_texts for _ in range(task_token_ids.shape[0])]
                        if upstream_texts is not None
                        else None,
                        max_new_tokens=max_new_tokens,
                        return_metadata=True,
                        inference_mode=inference_mode,
                        use_upstream_prefix=used_upstream_prefix,
                        do_sample=do_sample,
                    )
                    if collect_agent_logs:
                        agent_logs.append(
                            {
                                "agent_id": agents[j].agent_id,
                                "role_name": agents[j].role_name,
                                "output_type": "text",
                                "system_prompt": agents[j].system_prompt,
                                "received_upstream_prefix": upstream_prefix is not None,
                                "upstream_prefix": summarize_tensor("upstream_prefix", upstream_prefix),
                                "received_upstream_texts": bool(upstream_texts),
                                "upstream_texts": upstream_texts,
                                "inference_mode": inference_mode,
                                "used_upstream_prefix": used_upstream_prefix,
                                "generated_text": generation["generated_text"],
                                "generation": generation,
                            }
                        )

        if training:
             return {
                "final_logits": terminal_output["logits"],
                "question_len": terminal_output["question_len"],
                "answer_len": terminal_output["answer_len"],
                "all_prefixes": all_prefixes,
            }
        else:
            # Compute total generated tokens across all agents.
            # Latent mode: only terminal agent has generation → equals terminal count.
            # Pure-prefix mode: all agents have generation → sum of all agents' tokens.
            total_generated_tokens = None
            if agent_logs:
                per_agent_counts = []
                for log_entry in agent_logs:
                    gen = log_entry.get("generation")
                    if isinstance(gen, dict):
                        count = gen.get("generated_token_count", 0)
                        per_agent_counts.append(count)
                if per_agent_counts:
                    if any(isinstance(c, list) for c in per_agent_counts):
                        B = max(len(c) for c in per_agent_counts if isinstance(c, list))
                        total_generated_tokens = [
                            sum(
                                (c[b] if b < len(c) else 0) if isinstance(c, list) else int(c)
                                for c in per_agent_counts
                            )
                            for b in range(B)
                        ]
                    else:
                        total_generated_tokens = sum(int(c) for c in per_agent_counts)
            if total_generated_tokens is None:
                total_generated_tokens = generation.get("generated_token_count")

            return {
                "generated_text": generation["generated_text"],
                "generation": generation,
                "all_prefixes": all_prefixes,
                "agent_logs": agent_logs,
                "total_generated_tokens": total_generated_tokens,
            }
