# src/graph/dag_executor.py

import torch

from ..models.agent import Agent
from ..models.compressor import LatentCompressor
from ..communication.aggregator import MessageAggregator


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
        compressor: LatentCompressor,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
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
        communication_mode: str = "latent_prefix",
        text_message_edge_threshold: float = 0.5,
        text_message_max_new_tokens: int = 512,
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
        if training and communication_mode != "latent_prefix":
            raise ValueError("text_messages communication is only supported for inference/eval")

        all_prefixes = [None] * n
        all_text_messages = [None] * n
        agent_logs = []

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

        def collect_upstream_text_messages(agent_index: int) -> list[dict]:
            batch_size = task_token_ids.shape[0]
            upstream_messages_per_sample = [[] for _ in range(batch_size)]
            for i in execution_order:
                if i == agent_index:
                    break
                message_content = all_text_messages[i]
                if message_content is None:
                    continue
                edge_weight = float(adjacency[i, agent_index].detach().item())
                if edge_weight < text_message_edge_threshold:
                    continue
                if isinstance(message_content, list):
                    for sample_idx, sample_content in enumerate(message_content):
                        upstream_messages_per_sample[sample_idx].append(
                            {
                                "agent_id": agents[i].agent_id,
                                "role_name": agents[i].role_name,
                                "content": sample_content,
                                "edge_weight": edge_weight,
                            }
                        )
                else:
                    shared_message = {
                        "agent_id": agents[i].agent_id,
                        "role_name": agents[i].role_name,
                        "content": message_content,
                        "edge_weight": edge_weight,
                    }
                    for sample_messages in upstream_messages_per_sample:
                        sample_messages.append(dict(shared_message))
            if batch_size == 1:
                return upstream_messages_per_sample[0]
            return upstream_messages_per_sample

        for j in execution_order:
            upstream_prefix = None
            upstream_text_messages = []
            if communication_mode == "latent_prefix":
                # ── Step 1: Aggregate upstream prefixes ──
                upstream_prefix = self.aggregator.aggregate(
                    agent_index=j,
                    adjacency=adjacency,
                    all_prefixes=all_prefixes,
                )
            elif communication_mode == "text_messages":
                upstream_text_messages = collect_upstream_text_messages(j)
            else:
                raise ValueError(f"Unsupported communication_mode: {communication_mode}")

            if j != terminal_index:
                if communication_mode == "latent_prefix":
                    # ── Non-terminal: latent reasoning → compress → pass downstream ──
                    agent_output = agents[j].reason(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_prefix=upstream_prefix,
                    )
                    S_j = agent_output["hidden_trajectory"]
                    mask_j = agent_output["compressor_mask"]
                    P_j = compressor(S_j, mask=mask_j)
                    all_prefixes[j] = P_j
                    if collect_agent_logs:
                        agent_logs.append(
                            {
                                "agent_id": agents[j].agent_id,
                                "role_name": agents[j].role_name,
                                "output_type": "latent",
                                "system_prompt": agents[j].system_prompt,
                                "received_upstream_prefix": upstream_prefix is not None,
                                "upstream_prefix": summarize_tensor("upstream_prefix", upstream_prefix),
                                "hidden_trajectory": summarize_tensor("hidden_trajectory", S_j),
                                "compressed_prefix": summarize_tensor("compressed_prefix", P_j),
                            }
                        )
                else:
                    message_generation = agents[j].generate_message(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_text_messages=upstream_text_messages,
                        max_new_tokens=text_message_max_new_tokens,
                        return_metadata=True,
                        inference_mode=inference_mode,
                        do_sample=do_sample,
                    )
                    all_text_messages[j] = message_generation["generated_text"]
                    if collect_agent_logs:
                        agent_logs.append(
                            {
                                "agent_id": agents[j].agent_id,
                                "role_name": agents[j].role_name,
                                "output_type": "text_message",
                                "system_prompt": agents[j].system_prompt,
                                "upstream_text_messages": upstream_text_messages,
                                "generated_text": message_generation["generated_text"],
                                "generation": message_generation,
                            }
                        )
            else:
                # ── Terminal agent ──
                if training:
                    # Training: single forward pass, return logits for CE loss
                    terminal_output = agents[j].forward_for_loss(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_prefix=upstream_prefix,
                        answer_ids=answer_ids,
                        answer_mask=answer_mask,
                        input_mode=training_input_mode,
                    )
                else:
                    # Inference: autoregressive generation, return text
                    generation = agents[j].generate_answer(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_prefix=upstream_prefix,
                        upstream_text_messages=upstream_text_messages,
                        max_new_tokens=max_new_tokens,
                        return_metadata=True,
                        inference_mode=inference_mode,
                        use_upstream_prefix=(
                            use_terminal_prefix if communication_mode == "latent_prefix" else False
                        ),
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
                                "upstream_text_messages": upstream_text_messages,
                                "inference_mode": inference_mode,
                                "used_upstream_prefix": use_terminal_prefix,
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
            return {
                "generated_text": generation["generated_text"],
                "generation": generation,
                "all_prefixes": all_prefixes,
                "agent_logs": agent_logs,
            }
