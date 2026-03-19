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
        B = task_token_ids.shape[0]

        all_prefixes = []    # compressed P_i for each agent
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

        for j in range(n):
            # ── Step 1: Aggregate upstream prefixes ──
            upstream_prefix = self.aggregator.aggregate(
                agent_index=j,
                adjacency=adjacency,
                all_prefixes=all_prefixes,
            )

            if j < n - 1:
                # ── Non-terminal: latent reasoning → compress → pass downstream ──
                agent_output = agents[j].reason(
                    task_token_ids=task_token_ids,
                    task_attention_mask=task_attention_mask,
                    upstream_prefix=upstream_prefix,
                )
                S_j = agent_output["hidden_trajectory"]
                mask_j = agent_output["compressor_mask"]
                P_j = compressor(S_j, mask=mask_j)
                all_prefixes.append(P_j)
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
                all_prefixes.append(None)

                if training:
                    # Training: single forward pass, return logits for CE loss
                    terminal_output = agents[j].forward_for_loss(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_prefix=upstream_prefix,
                        answer_ids=answer_ids,
                        answer_mask=answer_mask,
                    )
                else:
                    # Inference: autoregressive generation, return text
                    generation = agents[j].generate_answer(
                        task_token_ids=task_token_ids,
                        task_attention_mask=task_attention_mask,
                        upstream_prefix=upstream_prefix,
                        max_new_tokens=max_new_tokens,
                        return_metadata=True,
                        inference_mode=inference_mode,
                        use_upstream_prefix=use_terminal_prefix,
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
