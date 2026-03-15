"""
BaseModelWrapper: loads and wraps a frozen HuggingFace causal LM.

Responsibilities:
  - Load model + tokenizer (from local cache or HF hub)
  - Freeze all parameters
  - Provide forward() that accepts optional prefix embeddings
  - Expose hidden_dim for downstream modules

Extensibility:
  - To support heterogeneous agents, instantiate multiple BaseModelWrapper
    with different model names. The interface stays the same.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModelWrapper(nn.Module):
    """Wrapper around a frozen HuggingFace causal language model."""

    def __init__(self, model_name: str, cache_dir: str = "./weights"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,  # use float32 for training stability
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze all parameters
        self._freeze()

        self._hidden_dim = self.model.config.hidden_size

    def _freeze(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def get_input_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Convert token IDs to embeddings without running the full model.

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
                Latent prefix from upstream agents, injected before input embeddings.
            output_hidden_states: whether to return all hidden states

        Returns:
            dict with keys:
                - logits: [batch_size, total_seq_len, vocab_size]
                - hidden_states: tuple of [batch_size, total_seq_len, hidden_dim]
                  (only if output_hidden_states=True)
                - last_hidden_state: [batch_size, total_seq_len, hidden_dim]
        """
        # Get token embeddings
        token_embeds = self.get_input_embeddings(input_ids)  # [B, S, D]

        # Prepend prefix if provided
        if prefix_embeds is not None:
            # prefix_embeds: [B, Lp, D], token_embeds: [B, S, D]
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # [B, Lp+S, D]

            # Extend attention mask for prefix tokens (always attended to)
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    attention_mask.shape[0],
                    prefix_embeds.shape[1],
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = token_embeds

        # Forward through the frozen model
        # Note: we pass inputs_embeds instead of input_ids
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        result = {
            "logits": outputs.logits,
            "last_hidden_state": outputs.hidden_states[-1] if output_hidden_states else None,
        }
        if output_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        return result

    def tokenize(self, texts: list[str], max_length: int = 512) -> dict:
        """Tokenize a batch of texts.

        Returns dict with input_ids and attention_mask.
        """
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
