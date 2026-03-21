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
        all_text_outputs = [None] * n
        agent_logs = []

        def collect_upstream_texts(agent_index: int) -> list[str]:
            texts = []
            for upstream_idx in range(n):
                message = all_text_outputs[upstream_idx]
                if message is None:
                    continue
                if float(adjacency[upstream_idx, agent_index].detach().item()) > 0.5:
                    texts.append(message)
            return texts

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

        for j in execution_order:
            # ── Step 1: Aggregate upstream prefixes ──
            upstream_prefix = self.aggregator.aggregate(
                agent_index=j,
                adjacency=adjacency,
                all_prefixes=all_prefixes,
            )

            if j != terminal_index:
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
                        agent_logs.append(
                            {
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
                            }
                        )
                    continue

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
                    upstream_texts = None
                    used_upstream_prefix = use_terminal_prefix
                    terminal_upstream_prefix = upstream_prefix
                    if inference_mode == "chat_with_text":
                        upstream_texts = collect_upstream_texts(j)
                        terminal_upstream_prefix = None
                        used_upstream_prefix = False
                    generation = agents[j].generate_answer(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_prefix=terminal_upstream_prefix,
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
            return {
                "generated_text": generation["generated_text"],
                "generation": generation,
                "all_prefixes": all_prefixes,
                "agent_logs": agent_logs,
            }
