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
        enable_thinking: bool = True,
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
        self.reasoning_steps = role_config.get("reasoning_steps", 256)
        self.base_model = base_model
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking
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

    @staticmethod
    def build_chat_prompt_text(
        tokenizer,
        question_text: str,
        system_prompt: str | None = None,
        enable_thinking: bool = True,
        upstream_text_messages: list[dict] | None = None,
        generation_purpose: str = "answer",
    ) -> str:
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append(
            {
                "role": "user",
                "content": Agent.build_chat_user_content(
                    question_text=question_text,
                    upstream_text_messages=upstream_text_messages,
                    generation_purpose=generation_purpose,
                ),
            }
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def format_upstream_text_messages(upstream_text_messages: list[dict] | None = None) -> str:
        if not upstream_text_messages:
            return ""

        lines = ["Upstream agent messages:"]
        for message in upstream_text_messages:
            role_name = str(message.get("role_name", "agent")).strip() or "agent"
            edge_weight = message.get("edge_weight")
            header = f"[{role_name}"
            if edge_weight is not None:
                header += f" | edge_weight={float(edge_weight):.3f}"
            header += "]"
            content = str(message.get("content", "")).strip() or "<empty>"
            lines.extend([header, content, ""])
        return "\n".join(lines).strip()

    @staticmethod
    def build_chat_user_content(
        question_text: str,
        upstream_text_messages: list[dict] | None = None,
        generation_purpose: str = "answer",
    ) -> str:
        question_text = question_text.strip()
        sections = [f"Task:\n{question_text}"]
        upstream_block = Agent.format_upstream_text_messages(upstream_text_messages)
        if upstream_block:
            sections.append(upstream_block)

        if generation_purpose == "message":
            sections.append(
                "Write a concise message for downstream agents. "
                "Preserve useful reasoning, avoid roleplay, and do not give the final answer unless it is unavoidable."
            )
        elif generation_purpose == "answer":
            sections.append(
                "Return the final answer. End with a standalone numeric answer when applicable."
            )
        else:
            raise ValueError(f"Unsupported generation_purpose: {generation_purpose}")

        return "\n\n".join(sections)

    def _build_generation_inputs(
        self,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        inference_mode: str = "legacy_plain_with_prefix",
        upstream_text_messages: list[list[dict]] | list[dict] | None = None,
        generation_purpose: str = "answer",
    ) -> tuple[torch.LongTensor, torch.Tensor | None]:
        if inference_mode == "legacy_plain_with_prefix":
            if upstream_text_messages:
                raise ValueError("upstream_text_messages require inference_mode='chat_with_prefix'")
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
            return input_ids, attention_mask

        if inference_mode != "chat_with_prefix":
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")

        question_texts = self.base_model.tokenizer.batch_decode(
            task_token_ids.detach().cpu(),
            skip_special_tokens=True,
        )
        batch_size = len(question_texts)
        if upstream_text_messages is None:
            upstream_batches = [None] * batch_size
        elif batch_size == 1 and isinstance(upstream_text_messages, list):
            upstream_batches = [upstream_text_messages]
        elif (
            isinstance(upstream_text_messages, list)
            and len(upstream_text_messages) == batch_size
            and all(message_group is None or isinstance(message_group, list) for message_group in upstream_text_messages)
        ):
            upstream_batches = upstream_text_messages
        else:
            upstream_batches = [upstream_text_messages] * batch_size
        prompts = [
            self.build_chat_prompt_text(
                tokenizer=self.base_model.tokenizer,
                question_text=question_text,
                system_prompt=self.system_prompt,
                enable_thinking=self.enable_thinking,
                upstream_text_messages=upstream_batches[idx],
                generation_purpose=generation_purpose,
            )
            for idx, question_text in enumerate(question_texts)
        ]
        tokenized = self.base_model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return input_ids, attention_mask

    @staticmethod
    def _left_pad_for_generation(
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        pad_token_id: int,
    ) -> tuple[torch.LongTensor, torch.Tensor | None]:
        if attention_mask is None:
            return input_ids, attention_mask

        valid_lengths = attention_mask.sum(dim=1)
        max_valid_len = int(valid_lengths.max().item())
        if torch.all(attention_mask == 1):
            return input_ids, attention_mask

        padded_input_ids = torch.full(
            (input_ids.shape[0], max_valid_len),
            pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        padded_attention_mask = torch.zeros(
            (attention_mask.shape[0], max_valid_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        for row_idx, valid_len in enumerate(valid_lengths.tolist()):
            valid_len = int(valid_len)
            valid_tokens = input_ids[row_idx][attention_mask[row_idx].bool()]
            padded_input_ids[row_idx, max_valid_len - valid_len:] = valid_tokens
            padded_attention_mask[row_idx, max_valid_len - valid_len:] = 1

        return padded_input_ids, padded_attention_mask

    @staticmethod
    def _infer_finish_reason(
        generated_ids: list[int],
        eos_token_id: int | None,
        max_new_tokens: int,
    ) -> str:
        if generated_ids and eos_token_id is not None and generated_ids[-1] == eos_token_id:
            return "eos"
        if len(generated_ids) >= max_new_tokens:
            return "max_new_tokens"
        return "stopped_early"

    @staticmethod
    def _finalize_generation_outputs(
        tokenizer,
        generated_token_ids: list[list[int]],
        eos_token_id: int | None,
        max_new_tokens: int,
        inference_mode: str,
        use_upstream_prefix: bool,
        return_metadata: bool,
    ) -> str | list[str] | dict:
        generated_texts = [
            tokenizer.decode(token_ids, skip_special_tokens=True)
            for token_ids in generated_token_ids
        ]
        finish_reasons = [
            Agent._infer_finish_reason(
                generated_ids=token_ids,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
            )
            for token_ids in generated_token_ids
        ]
        token_counts = [len(token_ids) for token_ids in generated_token_ids]
        stopped_early = [reason != "max_new_tokens" for reason in finish_reasons]

        if len(generated_texts) == 1:
            generated_text = generated_texts[0]
            finish_reason = finish_reasons[0]
            generated_token_count = token_counts[0]
            stopped = stopped_early[0]
        else:
            generated_text = generated_texts
            finish_reason = finish_reasons
            generated_token_count = token_counts
            stopped = stopped_early

        if return_metadata:
            return {
                "generated_text": generated_text,
                "finish_reason": finish_reason,
                "generated_token_count": generated_token_count,
                "stopped_early": stopped,
                "inference_mode": inference_mode,
                "used_upstream_prefix": use_upstream_prefix,
            }
        return generated_text

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
        input_mode: str = "legacy_plain_with_prefix",
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
        if input_mode == "legacy_plain_with_prefix":
            role_ids = self._get_role_token_ids().expand(task_token_ids.shape[0], -1)
            if answer_ids is not None:
                input_ids = torch.cat([role_ids, task_token_ids, answer_ids], dim=1)
            else:
                input_ids = torch.cat([role_ids, task_token_ids], dim=1)

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
            prompt_len = role_len + task_token_ids.shape[1]
            logits_start = role_len
        else:
            if input_mode != "chat_with_prefix":
                raise ValueError(f"Unsupported input_mode: {input_mode}")
            prompt_ids, prompt_mask = self._build_generation_inputs(
                task_token_ids=task_token_ids,
                task_attention_mask=task_attention_mask,
                inference_mode="chat_with_prefix",
            )
            if answer_ids is not None:
                input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            else:
                input_ids = prompt_ids
            if prompt_mask is not None and answer_ids is not None and answer_mask is not None:
                attention_mask = torch.cat([prompt_mask, answer_mask], dim=1)
            else:
                attention_mask = prompt_mask
            prompt_len = prompt_ids.shape[1]
            logits_start = 0

        # Forward through frozen model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_embeds=upstream_prefix,
            output_hidden_states=True,
        )

        # Keep [prompt ; answer] for chat mode and [question ; answer] for legacy mode,
        # so the masked labels still supervise the first answer token correctly.
        logits = outputs["logits"][:, logits_start:, :]

        return {
            "logits": logits,
            "question_len": prompt_len,
            "answer_len": answer_ids.shape[1] if answer_ids is not None else 0,
        }
        
    @torch.no_grad()
    def generate_answer(
        self,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        upstream_prefix: torch.Tensor | None = None,
        upstream_text_messages: list[dict] | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_metadata: bool = False,
        inference_mode: str = "legacy_plain_with_prefix",
        use_upstream_prefix: bool = True,
        generation_purpose: str = "answer",
    ) -> str | dict:
        input_ids, attention_mask = self._build_generation_inputs(
            task_token_ids=task_token_ids,
            task_attention_mask=task_attention_mask,
            inference_mode=inference_mode,
            upstream_text_messages=upstream_text_messages,
            generation_purpose=generation_purpose,
        )
        pad_token_id = self.base_model.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.base_model.tokenizer.eos_token_id or 0
        input_ids, attention_mask = self._left_pad_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
        )
        if not use_upstream_prefix:
            upstream_prefix = None
        eos_token_id = self.base_model.tokenizer.eos_token_id
        generation_model = (
            self.base_model._helper_model()
            if hasattr(self.base_model, "_helper_model")
            else self.base_model.model
        )

        # If we have upstream prefix, we need to:
        # 1. Convert input_ids to embeddings
        # 2. Prepend the prefix
        # 3. Delegate generation to HF so cache/position handling stays model-correct
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

            generate_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": pad_token_id,
                "return_dict_in_generate": True,
            }
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
            gen_out = generation_model.generate(**generate_kwargs)
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out[0]
            generated_ids = [sequences[row_idx].tolist() for row_idx in range(sequences.shape[0])]

            return self._finalize_generation_outputs(
                tokenizer=self.base_model.tokenizer,
                generated_token_ids=generated_ids,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                inference_mode=inference_mode,
                use_upstream_prefix=use_upstream_prefix,
                return_metadata=return_metadata,
            )

        else:
            # No prefix — just use model.generate directly
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.base_model.tokenizer.pad_token_id,
                "return_dict_in_generate": True,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p
            gen_out = generation_model.generate(
                **generate_kwargs,
            )
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out[0]
            generated_token_ids = [
                sequences[row_idx][input_ids.shape[1]:].tolist()
                for row_idx in range(sequences.shape[0])
            ]
            return self._finalize_generation_outputs(
                tokenizer=self.base_model.tokenizer,
                generated_token_ids=generated_token_ids,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens,
                inference_mode=inference_mode,
                use_upstream_prefix=use_upstream_prefix,
                return_metadata=return_metadata,
            )

    @torch.no_grad()
    def generate_message(
        self,
        task_token_ids: torch.LongTensor,
        task_attention_mask: torch.Tensor | None = None,
        upstream_text_messages: list[dict] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
        return_metadata: bool = False,
        inference_mode: str = "chat_with_prefix",
    ) -> str | dict:
        return self.generate_answer(
            task_token_ids=task_token_ids,
            task_attention_mask=task_attention_mask,
            upstream_prefix=None,
            upstream_text_messages=upstream_text_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_metadata=return_metadata,
            inference_mode=inference_mode,
            use_upstream_prefix=False,
            generation_purpose="message",
        )

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, role={self.role_name})"
