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
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModelWrapper(nn.Module):
    """Wrapper around a frozen HuggingFace causal language model."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        dtype: str | torch.dtype | None = None,
    ):
        super().__init__()

        # ── Load model & tokenizer ──
        load_path, load_kwargs = self._resolve_model_path(model_name, cache_dir)
        self.model = self._load_hf_model(
            load_path,
            load_kwargs,
            torch_dtype=self._resolve_dtype(dtype),
        )
        self.tokenizer = self._load_hf_tokenizer(load_path, load_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Freeze: requires_grad=False + permanent eval mode ──
        self._freeze()

        # Expose the full model config so downstream modules can read
        # hidden_size, num_attention_heads, etc. per model.
        self.model_config = self.model.config

    @staticmethod
    def _resolve_model_path(model_name: str, cache_dir: str | None) -> tuple[str, dict]:
        """Determine whether to load from a local directory or HF hub."""
        if cache_dir and os.path.isfile(os.path.join(cache_dir, "config.json")):
            return cache_dir, {}
        if os.path.isabs(model_name) and os.path.isfile(os.path.join(model_name, "config.json")):
            return model_name, {}
        if cache_dir:
            return model_name, {"cache_dir": cache_dir}
        return model_name, {}

    @staticmethod
    def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
        if dtype is None:
            return torch.float32
        if isinstance(dtype, torch.dtype):
            return dtype
        normalized = str(dtype).strip().lower()
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported model dtype: {dtype}")
        return mapping[normalized]

    @staticmethod
    def _load_hf_model(load_path: str, load_kwargs: dict, torch_dtype: torch.dtype):
        try:
            return AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **load_kwargs,
            )
        except Exception:
            offline_kwargs = dict(load_kwargs)
            offline_kwargs["local_files_only"] = True
            return AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **offline_kwargs,
            )

    @staticmethod
    def _load_hf_tokenizer(load_path: str, load_kwargs: dict):
        try:
            return AutoTokenizer.from_pretrained(
                load_path,
                trust_remote_code=True,
                **load_kwargs,
            )
        except Exception:
            offline_kwargs = dict(load_kwargs)
            offline_kwargs["local_files_only"] = True
            return AutoTokenizer.from_pretrained(
                load_path,
                trust_remote_code=True,
                **offline_kwargs,
            )

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
        self.base_model_trainable = False
        self.model.eval()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory when allowing
        gradients to flow through the frozen model forward pass."""
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def set_trainable(self, trainable: bool) -> None:
        self.base_model_trainable = trainable
        for param in self.model.parameters():
            param.requires_grad = trainable
        if trainable:
            self.model.train()
        else:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "base_model_trainable", False):
            self.model.train(mode)
        else:
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

    def _helper_model(self):
        """Return the underlying HF model for helper-style attribute access."""
        return self.model.module if hasattr(self.model, "module") else self.model
    
    @staticmethod
    def _parse_model_output(outputs, output_hidden_states: bool = True) -> dict:
        """Parse HF model output, handling both object and tuple formats."""
        if isinstance(outputs, tuple):
            parsed = {
                "logits": outputs[0],
                "past_key_values": outputs[1] if len(outputs) > 1 else None,
            }
            if output_hidden_states and len(outputs) > 2 and outputs[2] is not None:
                parsed["hidden_states"] = outputs[2]
            else:
                parsed["hidden_states"] = None
        else:
            parsed = {
                "logits": outputs.logits,
                "past_key_values": getattr(outputs, "past_key_values", None),
                "hidden_states": getattr(outputs, "hidden_states", None) if output_hidden_states else None,
            }
        return parsed

    @staticmethod
    def _past_length(past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        return int(past_key_values[0][0].shape[-2])

    def get_input_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, hidden_dim]
        """
        return self._helper_model().get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        prefix_embeds: torch.Tensor | None = None,
        past_key_values=None,
        output_hidden_states: bool = True,
    ) -> dict:
        if prefix_embeds is not None and past_key_values is not None:
            raise ValueError("prefix_embeds and past_key_values are mutually exclusive")

        if past_key_values is not None:
            prefix_len = self._past_length(past_key_values)
            if attention_mask is None:
                attention_mask = torch.ones(
                    input_ids.shape[0],
                    prefix_len + input_ids.shape[1],
                    device=input_ids.device,
                    dtype=torch.long,
                )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            raw = self._parse_model_output(outputs, output_hidden_states)
            full_logits = raw["logits"]
            full_last_hidden = raw["hidden_states"][-1] if raw["hidden_states"] else None

            return {
                "logits": full_logits,
                "last_hidden_state": full_last_hidden,
                "full_last_hidden_state": full_last_hidden,
                "prefix_len": prefix_len,
            }

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

        # Handle both object-style and tuple-style returns from HF models
        if isinstance(outputs, tuple):
            full_logits = outputs[0]
            # For tuple returns: (logits, past_kv, hidden_states, ...)
            # hidden_states is typically the 3rd element when output_hidden_states=True
            if output_hidden_states and len(outputs) > 2:
                full_last_hidden = outputs[2][-1] if outputs[2] is not None else None
            else:
                full_last_hidden = None
        else:
            raw = self._parse_model_output(outputs, output_hidden_states)
            full_logits = raw["logits"]
            full_last_hidden = raw["hidden_states"][-1] if raw["hidden_states"] else None

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
                    - generated_ids: [B, m]
                    - full_hidden: [B, input_len + m, D]
                    - prefix_len: int
            """
            B = input_ids.shape[0]
            device = input_ids.device

            # ── Step 1: Encode input (with optional prefix) ──
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

            # ── Initial forward to get KV cache ──
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            raw = self._parse_model_output(outputs, output_hidden_states=True)
            past_key_values = raw["past_key_values"]
            input_hidden = raw["hidden_states"][-1]       # [B, prefix+input_len, D]
            next_logits = raw["logits"][:, -1, :]         # [B, V]

            # ── Step 2: Auto-regressive generation ──
            hidden_trajectory = []
            generated_ids = []

            for step in range(max_new_tokens):
                # Sample next token
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
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
                raw = self._parse_model_output(outputs, output_hidden_states=True)
                past_key_values = raw["past_key_values"]
                step_hidden = raw["hidden_states"][-1]    # [B, 1, D]
                hidden_trajectory.append(step_hidden)
                next_logits = raw["logits"][:, -1, :]

                # Stop at EOS
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and (next_token == eos_id).all():
                    break

            # ── Stack results ──
            hidden_trajectory = torch.cat(hidden_trajectory, dim=1)  # [B, m, D]
            generated_ids = torch.cat(generated_ids, dim=1)          # [B, m]
            full_hidden = torch.cat([input_hidden, hidden_trajectory], dim=1)

            return {
                "hidden_trajectory": hidden_trajectory,
                "generated_ids": generated_ids,
                "full_hidden": full_hidden,
                "prefix_len": prefix_len,
            }
            
    @torch.amp.autocast("cuda", enabled=False)
    def compute_alignment_matrix(self, lambda_reg: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute input-output alignment matrix W_a (LatentMAS Eq. 3).

        W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in

        Also computes target_norm: the average norm of input embeddings,
        used to rescale aligned vectors to match input embedding scale.

        Computed ONCE and cached.

        Returns:
            (W_a, target_norm): alignment matrix [D, D] and scalar norm target
        """
        if hasattr(self, "_cached_alignment"):
            return self._cached_alignment

        helper_model = self._helper_model()
        W_in = helper_model.get_input_embeddings().weight.detach().float()  # [V, D]
        W_out = helper_model.lm_head.weight.detach().float()                # [V, D]

        gram = W_out.T @ W_out                                            # [D, D]
        gram += lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs = W_out.T @ W_in                                             # [D, D]
        W_a = torch.linalg.solve(gram, rhs)                              # [D, D]

        target_norm = W_in.norm(dim=1).mean()

        self._cached_alignment = (W_a, target_norm)
        return W_a, target_norm

    @torch.amp.autocast("cuda", enabled=False)
    def apply_alignment(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map last-layer hidden state back to input embedding space.

        Steps:
        1. h @ W_a  (project to input space)
        2. Rescale to match average input embedding norm

        Args:
            hidden: [B, D] or [B, 1, D] last-layer hidden state

        Returns:
            aligned: same shape as input, in input embedding space
        """
        W_a, target_norm = self.compute_alignment_matrix()
        W_a = W_a.to(device=hidden.device)
        target_norm = target_norm.to(device=hidden.device)

        h_float = hidden.float()
        aligned = torch.matmul(h_float, W_a)

        # Rescale to match input embedding norm
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm / aligned_norm)

        return aligned.to(hidden.dtype)

    def latent_reasoning(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        prefix_embeds: torch.Tensor | None = None,
        past_key_values=None,
        num_latent_steps: int = 40,
        grad_last_k: int = 0,
    ) -> dict:
        """Auto-regressive latent reasoning (LatentMAS Section 3.1).

        Each step:
        1. Take last-layer hidden state h_t from previous step
        2. Align: e_{t+1} = apply_alignment(h_t)
        3. Feed e_{t+1} as inputs_embeds with KV cache
        4. Get new h_{t+1}, repeat

        No tokens are decoded. Reasoning is entirely in latent space.

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len] or None
            prefix_embeds: [B, Lp, D] from upstream agents, or None
            num_latent_steps: m, number of latent reasoning steps
            grad_last_k: number of final steps to keep gradient graph for.
                         0 = all steps follow the outer context (no_grad or not).
                         >0 = first (m - grad_last_k) steps use no_grad,
                               last grad_last_k steps keep gradient graph.

        Returns:
            dict with:
                - hidden_trajectory: [B, m, D] the m latent thoughts
                - prefix_len: int
        """
        B = input_ids.shape[0]
        device = input_ids.device

        if prefix_embeds is not None and past_key_values is not None:
            raise ValueError("prefix_embeds and past_key_values are mutually exclusive")

        # ── Step 1: Encode input (with optional prefix or prebuilt KV cache) ──
        if past_key_values is not None:
            prefix_len = self._past_length(past_key_values)
            if attention_mask is None:
                attention_mask = torch.ones(
                    B,
                    prefix_len + input_ids.shape[1],
                    device=device,
                    dtype=torch.long,
                )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
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

            if attention_mask is None:
                attention_mask = torch.ones(B, inputs_embeds.shape[1], device=device, dtype=torch.long)

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        raw = self._parse_model_output(outputs, output_hidden_states=True)
        past_kv = raw["past_key_values"]
        last_hidden = raw["hidden_states"][-1][:, -1, :]  # [B, D]

        # Save full hidden states from initial encoding for the compressor.
        # KV cache path: prefix is in the cache, so output hidden states cover
        # only input_ids positions — take them as-is.
        # Embedding path: prefix is prepended to inputs_embeds, so slice it off.
        if past_key_values is not None:
            initial_hidden = raw["hidden_states"][-1]          # [B, input_len, D]
        else:
            initial_hidden = raw["hidden_states"][-1][:, prefix_len:, :]  # [B, input_len, D]

        # ── Step 2: Latent reasoning loop ──
        hidden_trajectory = []
        # Determine which steps need gradient
        grad_start = num_latent_steps - grad_last_k if grad_last_k > 0 else -1

        for step in range(num_latent_steps):
            # Align h_t to input embedding space
            aligned = self.apply_alignment(last_hidden)     # [B, D]
            latent_embed = aligned.unsqueeze(1)              # [B, 1, D]

            # Build attention mask covering all past + new position
            past_len = self._past_length(past_kv)
            latent_mask = torch.ones(
                B, past_len + 1,
                dtype=torch.long,
                device=device,
            )

            # Early steps: no_grad (pure inference, save memory)
            # Last grad_last_k steps: keep gradient graph
            use_no_grad = grad_last_k > 0 and step < grad_start

            if use_no_grad:
                with torch.no_grad():
                    outputs = self.model(
                        inputs_embeds=latent_embed,
                        attention_mask=latent_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    raw = self._parse_model_output(outputs, output_hidden_states=True)
                    past_kv = raw["past_key_values"]
                    last_hidden = raw["hidden_states"][-1][:, -1, :]
                    hidden_trajectory.append(last_hidden.unsqueeze(1).detach())
            else:
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                raw = self._parse_model_output(outputs, output_hidden_states=True)
                past_kv = raw["past_key_values"]
                last_hidden = raw["hidden_states"][-1][:, -1, :]
                hidden_trajectory.append(last_hidden.unsqueeze(1))

        # Stack: [B, m, D]
        hidden_trajectory = torch.cat(hidden_trajectory, dim=1)

        return {
            "hidden_trajectory": hidden_trajectory,
            "initial_hidden": initial_hidden,
            "prefix_len": prefix_len,
        }

    def tokenize(
        self,
        texts: list[str],
        max_length: int = 512,
        add_special_tokens: bool = True,
    ) -> dict:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        
    def _find_user_start(self, input_ids, pattern_ids):
        pattern = torch.tensor(pattern_ids, device=input_ids.device)
        pattern_len = len(pattern_ids)
        positions = []
        for i in range(input_ids.shape[0]):
            found = False
            for j in range(input_ids.shape[1] - pattern_len + 1):
                if torch.equal(input_ids[i, j:j+pattern_len], pattern):
                    positions.append(j + pattern_len)
                    found = True
                    break
            if not found:
                positions.append(0)  # fallback: prepend
        return positions
