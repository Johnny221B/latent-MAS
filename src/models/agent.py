# src/models/agent.py
"""
Agent: a role-aware wrapper around a base model.

Each Agent:
  - Holds a reference to a BaseModelWrapper (does NOT own the weights)
  - Has a role config (loaded from JSON)
  - Performs m-step latent reasoning given task input + optional upstream prefix
  - Returns hidden-state trajectory S_i for compression

Extensibility:
  - For heterogeneous agents, pass different BaseModelWrapper instances.
  - For different reasoning strategies, subclass and override `reason()`.
"""

import torch
from .base_model import BaseModelWrapper


class Agent:
    """A role-aware agent that performs latent reasoning using a shared base model."""

    def __init__(
        self,
        agent_id: int,
        role_config: dict,
        base_model: BaseModelWrapper,
        max_seq_len: int = 512,
    ):
        """
        Args:
            agent_id: unique integer index in the graph
            role_config: dict loaded from role JSON file, must contain:
                - role_name: str
                - system_prompt: str
                - reasoning_steps: int (m)
            base_model: shared BaseModelWrapper instance
            max_seq_len: maximum sequence length for tokenization
        """
        self.agent_id = agent_id
        self.role_config = role_config
        self.role_name = role_config["role_name"]
        self.system_prompt = role_config["system_prompt"]
        self.reasoning_steps = role_config.get("reasoning_steps", 4)
        self.base_model = base_model
        self.max_seq_len = max_seq_len
        self.reasoning_steps = role_config.get("reasoning_steps", 256)
        self.compress_last_k = role_config.get("compress_last_k", 40)

        # Pre-tokenize the role prompt (system prompt)
        self._role_tokens = None  # lazily initialized

    @property
    def device(self) -> torch.device:
        return self.base_model.device

    def _get_role_token_ids(self) -> torch.LongTensor:
        """Tokenize the system prompt once and cache it."""
        if self._role_tokens is None:
            encoded = self.base_model.tokenizer(
                self.system_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            self._role_tokens = encoded["input_ids"]  # [1, role_len]
        return self._role_tokens.to(self.device)

    def build_input_ids(self, task_token_ids: torch.LongTensor) -> torch.LongTensor:
        """Construct input sequence: [role_prompt_tokens; task_tokens].

        The upstream prefix (if any) is NOT included here — it will be
        prepended as embeddings in the forward pass.

        Args:
            task_token_ids: [batch_size, task_seq_len]

        Returns:
            input_ids: [batch_size, role_len + task_seq_len]
        """
        B = task_token_ids.shape[0]
        role_ids = self._get_role_token_ids().expand(B, -1)  # [B, role_len]
        return torch.cat([role_ids, task_token_ids], dim=1)

    def reason(
        self,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        upstream_prefix: torch.Tensor | None = None,
    ) -> dict:
        """Perform reasoning via auto-regressive generation and return hidden trajectory.

        The agent actually "thinks" by generating tokens. The hidden states
        produced during generation form the reasoning trace S_i.

        Returns:
            dict with:
                - hidden_trajectory: [B, m, D] reasoning trace for compressor
                - full_hidden: [B, input_len + m, D] complete hidden states
                - logits: not available here (use generated_ids for eval)
                - generated_ids: [B, m] generated tokens (for debugging)
                - compressor_mask: [B, m] mask for hidden_trajectory (all 1s, no padding)
        """
        # Build input: [role_prompt; task]
        input_ids = self.build_input_ids(task_token_ids)

        # Build attention mask
        if task_attention_mask is not None:
            role_mask = torch.ones(
                task_attention_mask.shape[0],
                self._get_role_token_ids().shape[1],
                device=task_attention_mask.device,
                dtype=task_attention_mask.dtype,
            )
            attention_mask = torch.cat([role_mask, task_attention_mask], dim=1)
        else:
            attention_mask = None

        # Generate with hidden state collection
        gen_output = self.base_model.generate_with_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=upstream_prefix,
            max_new_tokens=self.reasoning_steps,
        )

        trajectory = gen_output["hidden_trajectory"]  # [B, m, D]
        B, m, D = trajectory.shape

        # Only compress the last k hidden states — these contain the most
        # condensed reasoning since each position attends to all prior ones.
        k = min(self.compress_last_k, m)  # handle case where generation < k
        trajectory_to_compress = trajectory[:, -k:, :]  # [B, k, D]
        compressor_mask = torch.ones(B, k, device=task_token_ids.device)

        return {
            "hidden_trajectory": trajectory_to_compress,  # [B, k, D] for compressor
            "full_hidden": gen_output["full_hidden"],
            "full_trajectory": trajectory,                 # [B, m, D] keep full for logging
            "compressor_mask": compressor_mask,
            "generated_ids": gen_output["generated_ids"],
            "prefix_len": gen_output["prefix_len"],
        }
        
    def forward_for_loss(
        self,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        upstream_prefix: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass for terminal agent — returns logits for CE loss.

        Unlike reason(), this does NOT generate tokens. It runs a single
        forward pass and returns text-aligned logits for loss computation.
        """
        input_ids = self.build_input_ids(task_token_ids)

        if task_attention_mask is not None:
            role_mask = torch.ones(
                task_attention_mask.shape[0],
                self._get_role_token_ids().shape[1],
                device=task_attention_mask.device,
                dtype=task_attention_mask.dtype,
            )
            attention_mask = torch.cat([role_mask, task_attention_mask], dim=1)
        else:
            attention_mask = None

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=upstream_prefix,
            output_hidden_states=False,
        )

        role_len = self._get_role_token_ids().shape[1]

        return {
            "logits": outputs["logits"][:, role_len:, :],  # task-only logits
        }

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, role={self.role_name})"