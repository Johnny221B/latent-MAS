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
            """Perform latent reasoning and return hidden trajectory.

            Each agent reasons in continuous latent space for m steps,
            then returns the last k hidden states for compression.

            Returns:
                dict with:
                    - hidden_trajectory: [B, k, D] last k latent thoughts for compressor
                    - compressor_mask: [B, k] all ones (no padding in latent thoughts)
                    - full_trajectory: [B, m, D] complete latent thoughts (for logging)
                    - prefix_len: int
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

            # Latent reasoning in continuous space (no grad needed — frozen model)
            with torch.no_grad():
                output = self.base_model.latent_reasoning(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prefix_embeds=upstream_prefix,
                    num_latent_steps=self.reasoning_steps,
                )

            # Detach trajectory but keep it as a regular tensor for compressor
            # Compressor will create its own computation graph from here
            output["hidden_trajectory"] = output["hidden_trajectory"].detach()

            trajectory = output["hidden_trajectory"]  # [B, m, D]
            B, m, D = trajectory.shape

            # Only compress the last k hidden states
            k = min(self.compress_last_k, m)
            trajectory_to_compress = trajectory[:, -k:, :]
            compressor_mask = torch.ones(B, k, device=trajectory.device)

            return {
                "hidden_trajectory": trajectory_to_compress,
                "compressor_mask": compressor_mask,
                "full_trajectory": trajectory,
                "prefix_len": output["prefix_len"],
            }
        
    def forward_for_loss(
        self,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        upstream_prefix: torch.Tensor | None = None,
        answer_ids: torch.LongTensor | None = None,
        answer_mask: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass for terminal agent with teacher forcing.

        Input to model: [prefix ; role_prompt ; question ; answer]
        Logits returned: aligned with [question ; answer] (role sliced off)
        Labels should mask question positions with -100, keep answer positions.

        Args:
            task_token_ids: [B, question_len]
            task_attention_mask: [B, question_len]
            upstream_prefix: [B, Lp, D] or None
            answer_ids: [B, answer_len] ground truth answer tokens
            answer_mask: [B, answer_len] attention mask for answer
        """
        # Build: [role_prompt ; question ; answer]
        role_ids = self._get_role_token_ids().expand(task_token_ids.shape[0], -1)
        
        if answer_ids is not None:
            input_ids = torch.cat([role_ids, task_token_ids, answer_ids], dim=1)
        else:
            input_ids = torch.cat([role_ids, task_token_ids], dim=1)

        # Build attention mask
        B = task_token_ids.shape[0]
        role_len = role_ids.shape[1]
        role_mask_part = torch.ones(B, role_len, device=task_token_ids.device, dtype=torch.long)

        if task_attention_mask is not None:
            if answer_ids is not None and answer_mask is not None:
                attention_mask = torch.cat([role_mask_part, task_attention_mask, answer_mask], dim=1)
            else:
                attention_mask = torch.cat([role_mask_part, task_attention_mask], dim=1)
        else:
            attention_mask = None

        # Forward through frozen model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=upstream_prefix,
            output_hidden_states=True,
        )

        # Slice off role_prompt positions, keep [question ; answer]
        logits = outputs["logits"][:, role_len:, :]

        return {
            "logits": logits,                # [B, question_len + answer_len, V]
            "question_len": task_token_ids.shape[1],
            "answer_len": answer_ids.shape[1] if answer_ids is not None else 0,
        }
        
    @torch.no_grad()
    def generate_answer(
        self,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        upstream_prefix: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        input_ids = self.build_input_ids(task_token_ids)  # [1, role_len + task_len]

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

        # If we have upstream prefix, we need to:
        # 1. Convert input_ids to embeddings
        # 2. Prepend the prefix
        # 3. Do a forward pass to get KV cache
        # 4. Then generate from there
        if upstream_prefix is not None:
            token_embeds = self.base_model.get_input_embeddings(input_ids)
            prefix = upstream_prefix.to(dtype=token_embeds.dtype, device=token_embeds.device)
            inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

            if attention_mask is not None:
                prefix_mask = torch.ones(
                    attention_mask.shape[0],
                    prefix.shape[1],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # Forward pass to build KV cache from prefix + input
            outputs = self.base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values if hasattr(outputs, 'past_key_values') else outputs[1]

            # Now generate token by token using KV cache
            generated_ids = []
            raw = self.base_model._parse_model_output(outputs, output_hidden_states=False)
            next_logits = raw["logits"][:, -1, :]  # [1, V]

            for step in range(max_new_tokens):
                if do_sample and temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = sorted_indices.gather(1, torch.multinomial(sorted_probs, 1))
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated_ids.append(next_token.item())

                if next_token.item() == self.base_model.tokenizer.eos_token_id:
                    break

                # Forward next token with KV cache
                next_embed = self.base_model.get_input_embeddings(next_token)

                if hasattr(past_kv, "get_seq_length"):
                    past_len = past_kv.get_seq_length()
                else:
                    past_len = past_kv[0][0].shape[-2]

                step_mask = torch.ones(1, past_len + 1, dtype=torch.long, device=input_ids.device)
                out = self.base_model.model(
                    inputs_embeds=next_embed,
                    attention_mask=step_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                raw = self.base_model._parse_model_output(out, output_hidden_states=False)
                past_kv = raw["past_key_values"]
                next_logits = raw["logits"][:, -1, :]

            return self.base_model.tokenizer.decode(generated_ids, skip_special_tokens=True)

        else:
            # No prefix — just use model.generate directly
            gen_out = self.base_model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.base_model.tokenizer.pad_token_id,
            )
            return self.base_model.tokenizer.decode(
                gen_out[0][input_ids.shape[1]:], skip_special_tokens=True
            )

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, role={self.role_name})"