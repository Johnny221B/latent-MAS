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
    ) -> torch.Tensor:
        """Perform latent reasoning and return hidden-state trajectory.

        This is the core computation of an agent:
        1. Build input: [role_prompt; task_description]
        2. Prepend upstream_prefix as latent embeddings (if any)
        3. Run forward through frozen base model
        4. Extract hidden states as the reasoning trajectory S_i

        Args:
            task_token_ids: [batch_size, task_seq_len]
            task_attention_mask: [batch_size, task_seq_len] or None
            upstream_prefix: [batch_size, Lp, hidden_dim] or None
                Aggregated latent prefix from upstream agents.

        Returns:
            S_i: [batch_size, output_seq_len, hidden_dim]
                The hidden-state trajectory representing this agent's reasoning.
                output_seq_len = (prefix_len) + role_len + task_seq_len
        """
        # Build the text part of input
        input_ids = self.build_input_ids(task_token_ids)  # [B, role_len + task_len]

        # Build attention mask for the text part
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

        # Forward pass through frozen model with optional prefix
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=upstream_prefix,
            output_hidden_states=True,
        )

        # Return the last hidden state as the reasoning trajectory S_i
        # Shape: [B, total_seq_len, D] where total = prefix_len + role_len + task_len
        return outputs["last_hidden_state"]

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, role={self.role_name})"
