# src/models/base_model.py
"""
BaseModelWrapper: loads and wraps a frozen HuggingFace causal LM.

Responsibilities:
  - Load model + tokenizer (from local cache or HF hub)
  - Freeze all parameters AND lock eval mode (dropout stays off)
  - Provide forward() that accepts optional prefix embeddings
  - Return prefix-aware outputs: slice off prefix positions so downstream
    modules see logits/hidden_states aligned with the text tokens only
  - Expose model_config for downstream modules to read hidden_dim, num_heads, etc.

Extensibility:
  - To support heterogeneous agents, instantiate multiple BaseModelWrapper
    with different model names. Downstream modules should read hidden_dim
    from wrapper.model_config rather than assuming a shared constant.
"""

import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModelWrapper(nn.Module):
    """Wrapper around a frozen HuggingFace causal language model."""

    def __init__(self, model_name: str, cache_dir: str = "./weights"):
        super().__init__()

        # ── Load model & tokenizer ──
        load_path, load_kwargs = self._resolve_model_path(model_name, cache_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            **load_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=True,
            **load_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Freeze: requires_grad=False + permanent eval mode ──
        self._freeze()

        # Expose the full model config so downstream modules can read
        # hidden_size, num_attention_heads, etc. per model.
        self.model_config = self.model.config

    @staticmethod
    def _resolve_model_path(model_name: str, cache_dir: str) -> tuple[str, dict]:
        """Determine whether to load from a local directory or HF hub."""
        if os.path.isfile(os.path.join(cache_dir, "config.json")):
            return cache_dir, {}
        if os.path.isabs(model_name) and os.path.isfile(os.path.join(model_name, "config.json")):
            return model_name, {}
        return model_name, {"cache_dir": cache_dir}

    def _freeze(self):
        """Freeze all model parameters and lock to eval mode.

        Two things are needed:
          1. requires_grad = False  → no gradient computation
          2. model.eval()           → dropout / batchnorm stay in inference mode

        We also override train() so that even if the top-level
        MultiAgentSystem.train() is called, this model stays in eval mode.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension of this specific base model.

        Downstream modules (compressor, agent) should read this per-wrapper
        rather than assuming a global constant, to support heterogeneous setups.
        """
        return self.model_config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def get_input_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, hidden_dim]
        """
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        prefix_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> dict:
        """Forward pass with optional prefix embeddings prepended.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or None
            prefix_embeds: [batch_size, prefix_len, hidden_dim] or None
            output_hidden_states: whether to return all hidden states

        Returns:
            dict with keys:
                - logits: [B, text_seq_len, vocab_size]
                    Logits for text tokens ONLY (prefix positions sliced off).
                - last_hidden_state: [B, text_seq_len, hidden_dim]
                    Last layer hidden states for text tokens ONLY.
                - full_last_hidden_state: [B, prefix_len + text_seq_len, hidden_dim]
                    Full hidden states INCLUDING prefix positions.
                    (Needed by compressor which operates on the full trajectory.)
                - prefix_len: int, number of prefix tokens (0 if no prefix).
        """
        # Get token embeddings
        token_embeds = self.get_input_embeddings(input_ids)  # [B, S, D]

        prefix_len = 0

        if prefix_embeds is not None:
            prefix_embeds = prefix_embeds.to(
                dtype=token_embeds.dtype,
                device=token_embeds.device,
            )
            prefix_len = prefix_embeds.shape[1]

            # Concat: [prefix; text_tokens]
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

            # Extend attention mask for prefix (always fully attended)
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    attention_mask.shape[0],
                    prefix_len,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = token_embeds

        # Forward through frozen model (always in eval mode)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        full_logits = outputs.logits                    # [B, Lp+S, V]
        full_last_hidden = (
            outputs.hidden_states[-1] if output_hidden_states else None
        )

        text_logits = full_logits[:, prefix_len:, :]            # [B, S, V]
        text_hidden = (
            full_last_hidden[:, prefix_len:, :] if full_last_hidden is not None else None
        )

        return {
            "logits": text_logits,                    # text-only, for task loss
            "last_hidden_state": text_hidden,         # text-only
            "full_last_hidden_state": full_last_hidden,  # full, for compressor
            "prefix_len": prefix_len,
        }
        
    @torch.no_grad()
    def generate_with_hidden(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        prefix_embeds: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> dict:
        """Auto-regressive generation that collects hidden states at each step.

        This is the core of agent reasoning: the LLM actually "thinks" by
        generating tokens, and we capture the hidden trajectory.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len] or None
            prefix_embeds: [B, Lp, D] or None
            max_new_tokens: number of reasoning tokens to generate (m)
            temperature: sampling temperature
            top_p: nucleus sampling threshold

        Returns:
            dict with:
                - hidden_trajectory: [B, m, D]
                    Last-layer hidden state at each generation step.
                    This is the reasoning trace S_i for compression.
                - generated_ids: [B, m]
                    The token IDs generated (for debugging / logging).
                - full_hidden: [B, input_len + m, D]
                    Complete hidden states including input encoding.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # Step 1: Encode input (with optional prefix)
        token_embeds = self.get_input_embeddings(input_ids)
        prefix_len = 0

        if prefix_embeds is not None:
            prefix_embeds = prefix_embeds.to(dtype=token_embeds.dtype, device=device)
            prefix_len = prefix_embeds.shape[1]
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(B, prefix_len, device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = token_embeds

        # Initial forward to get KV cache
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values

        # Collect the input encoding hidden states
        input_hidden = outputs.hidden_states[-1]  # [B, prefix+input_len, D]

        # Step 2: Auto-regressive generation, collecting hidden states
        hidden_trajectory = []
        generated_ids = []

        # Start from the last token's logits
        next_logits = outputs.logits[:, -1, :]  # [B, V]

        for step in range(max_new_tokens):
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                # Top-p filtering
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_indices.gather(1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated_ids.append(next_token)  # [B, 1]

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(B, 1, device=device, dtype=attention_mask.dtype),
                ], dim=1)

            # Forward with KV cache (only the new token)
            next_embeds = self.get_input_embeddings(next_token)  # [B, 1, D]
            outputs = self.model(
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )

            past_key_values = outputs.past_key_values
            step_hidden = outputs.hidden_states[-1]  # [B, 1, D]
            hidden_trajectory.append(step_hidden)
            next_logits = outputs.logits[:, -1, :]

            # Stop at EOS
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and (next_token == eos_id).all():
                break

        # Stack results
        hidden_trajectory = torch.cat(hidden_trajectory, dim=1)  # [B, m, D]
        generated_ids = torch.cat(generated_ids, dim=1)          # [B, m]
        full_hidden = torch.cat([input_hidden, hidden_trajectory], dim=1)

        return {
            "hidden_trajectory": hidden_trajectory,  # reasoning trace for compressor
            "generated_ids": generated_ids,           # for debugging
            "full_hidden": full_hidden,               # complete trajectory
            "prefix_len": prefix_len,
        }

    def tokenize(self, texts: list[str], max_length: int = 512) -> dict:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
