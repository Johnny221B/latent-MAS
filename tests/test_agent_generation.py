import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.agent import Agent


class _FakeCache:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def get_seq_length(self) -> int:
        return self.seq_len


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.enable_thinking_values = []
        self.messages_history = []

    def __call__(self, prompts, return_tensors=None, padding=False, truncation=False, max_length=None, add_special_tokens=False, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        input_ids = torch.tensor([[5, 6], [7, 8]], dtype=torch.long)[:batch_size]
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [f"question-{idx}" for idx in range(token_ids.shape[0])]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        self.enable_thinking_values.append(enable_thinking)
        self.messages_history.append(messages)
        return messages[-1]["content"]

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "|".join(str(x) for x in token_ids)


class _FakeModel:
    def __init__(self, vocab_size=10, hidden_dim=4):
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim
        self.attention_masks = []
        self.generate_attention_mask = None
        self.generate_inputs_embeds_shape = None

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, inputs_embeds=None, attention_mask=None, use_cache=False, return_dict=True, past_key_values=None, **kwargs):
        batch_size, seq_len, _ = inputs_embeds.shape
        logits = torch.zeros(batch_size, seq_len, 10)
        next_token_id = 1 if past_key_values is None else 2
        logits[:, -1, next_token_id] = 10.0
        if attention_mask is not None:
            self.attention_masks.append(attention_mask.detach().clone())
        cache_len = attention_mask.shape[1] if attention_mask is not None else seq_len
        return type(
            "FakeOutput",
            (),
            {
                "logits": logits,
                "past_key_values": _FakeCache(cache_len),
            },
        )()

    def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=None, do_sample=None, pad_token_id=None, return_dict_in_generate=None, **kwargs):
        if inputs_embeds is not None:
            self.generate_inputs_embeds_shape = tuple(inputs_embeds.shape)
        if attention_mask is not None:
            self.generate_attention_mask = attention_mask.detach().clone()
        input_ids = kwargs.get("input_ids")
        batch_size = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
        sequences = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)[:batch_size]
        return type("FakeGenerateOutput", (), {"sequences": sequences})()


class _FakeBaseModel:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()

    @property
    def device(self):
        return torch.device("cpu")

    def _helper_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def get_input_embeddings(self, input_ids):
        return self._helper_model().get_input_embeddings()(input_ids)

    @staticmethod
    def _parse_model_output(outputs, output_hidden_states=False):
        return {
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": None,
        }


class _FakeDDP:
    def __init__(self, module):
        self.module = module


def test_generate_answer_with_prefix_supports_batched_inputs():
    agent = Agent(
        agent_id=0,
        role_config={
            "role_name": "summarizer",
            "system_prompt": "You summarize.",
            "reasoning_steps": 4,
            "compress_last_k": 4,
        },
        base_model=_FakeBaseModel(),
    )

    result = agent.generate_answer(
        task_token_ids=torch.tensor([[11, 12], [13, 14]], dtype=torch.long),
        task_attention_mask=torch.ones(2, 2, dtype=torch.long),
        upstream_prefix=torch.zeros(2, 1, 4),
        max_new_tokens=3,
        do_sample=False,
        return_metadata=True,
        inference_mode="chat_with_prefix",
    )

    assert result["generated_text"] == ["1|2", "1|2"]
    assert result["finish_reason"] == ["eos", "eos"]
    assert result["generated_token_count"] == [2, 2]
    assert result["stopped_early"] == [True, True]
    assert agent.base_model.model.generate_inputs_embeds_shape == (2, 3, 4)
    assert torch.equal(
        agent.base_model.model.generate_attention_mask,
        torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
    )
    assert agent.base_model.tokenizer.enable_thinking_values == [True, True]


def test_generate_answer_with_prefix_preserves_padding_mask_across_steps():
    base_model = _FakeBaseModel()
    agent = Agent(
        agent_id=0,
        role_config={
            "role_name": "summarizer",
            "system_prompt": "You summarize.",
            "reasoning_steps": 4,
            "compress_last_k": 4,
        },
        base_model=base_model,
    )

    agent.generate_answer(
        task_token_ids=torch.tensor([[3, 4], [5, 0]], dtype=torch.long),
        task_attention_mask=torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
        upstream_prefix=torch.zeros(2, 1, 4),
        max_new_tokens=2,
        do_sample=False,
        return_metadata=False,
        inference_mode="legacy_plain_with_prefix",
    )

    assert torch.equal(
        base_model.model.generate_attention_mask,
        torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 1, 1]], dtype=torch.long),
    )


def test_generate_answer_uses_unwrapped_model_for_ddp_generate():
    base_model = _FakeBaseModel()
    wrapped_model = _FakeModel()
    base_model.model = _FakeDDP(wrapped_model)
    agent = Agent(
        agent_id=0,
        role_config={
            "role_name": "summarizer",
            "system_prompt": "You summarize.",
            "reasoning_steps": 4,
            "compress_last_k": 4,
        },
        base_model=base_model,
    )

    result = agent.generate_answer(
        task_token_ids=torch.tensor([[11, 12]], dtype=torch.long),
        task_attention_mask=torch.ones(1, 2, dtype=torch.long),
        upstream_prefix=torch.zeros(1, 1, 4),
        max_new_tokens=3,
        do_sample=False,
        return_metadata=True,
        inference_mode="chat_with_prefix",
    )

    assert result["generated_text"] == "1|2"
    assert wrapped_model.generate_inputs_embeds_shape == (1, 3, 4)


def test_forward_for_loss_chat_mode_enables_thinking_by_default():
    base_model = _FakeBaseModel()
    agent = Agent(
        agent_id=0,
        role_config={
            "role_name": "summarizer",
            "system_prompt": "You summarize.",
            "reasoning_steps": 4,
            "compress_last_k": 4,
        },
        base_model=base_model,
    )

    class _FakeForwardBaseModel(_FakeBaseModel):
        def __call__(self, input_ids=None, attention_mask=None, prefix_embeds=None, output_hidden_states=False):
            batch_size, seq_len = input_ids.shape
            return {"logits": torch.zeros(batch_size, seq_len, 10)}

    forward_model = _FakeForwardBaseModel()
    agent.base_model = forward_model

    agent.forward_for_loss(
        task_token_ids=torch.tensor([[11, 12]], dtype=torch.long),
        task_attention_mask=torch.ones(1, 2, dtype=torch.long),
        upstream_prefix=None,
        answer_ids=torch.tensor([[3, 4]], dtype=torch.long),
        answer_mask=torch.ones(1, 2, dtype=torch.long),
        input_mode="chat_with_prefix",
    )

    assert forward_model.tokenizer.enable_thinking_values == [True]


def test_generate_answer_can_disable_thinking_explicitly():
    base_model = _FakeBaseModel()
    agent = Agent(
        agent_id=0,
        role_config={
            "role_name": "summarizer",
            "system_prompt": "You summarize.",
            "reasoning_steps": 4,
            "compress_last_k": 4,
        },
        base_model=base_model,
        enable_thinking=False,
    )

    agent.generate_answer(
        task_token_ids=torch.tensor([[11, 12]], dtype=torch.long),
        task_attention_mask=torch.ones(1, 2, dtype=torch.long),
        upstream_prefix=torch.zeros(1, 1, 4),
        max_new_tokens=3,
        do_sample=False,
        return_metadata=True,
        inference_mode="chat_with_prefix",
    )

    assert base_model.tokenizer.enable_thinking_values == [False]


def test_generate_answer_chat_mode_includes_upstream_text_messages_in_template():
    base_model = _FakeBaseModel()
    agent = Agent(
        agent_id=1,
        role_config={
            "role_name": "planner",
            "system_prompt": "You plan.",
            "reasoning_steps": 4,
            "compress_last_k": 4,
        },
        base_model=base_model,
    )

    agent.generate_answer(
        task_token_ids=torch.tensor([[11, 12]], dtype=torch.long),
        task_attention_mask=torch.ones(1, 2, dtype=torch.long),
        upstream_prefix=None,
        upstream_text_messages=[
            {"role_name": "reader", "content": "Focus on the egg counts.", "edge_weight": 0.82},
            {"role_name": "solver", "content": "Subtract breakfast and muffins.", "edge_weight": 0.81},
        ],
        max_new_tokens=3,
        do_sample=False,
        return_metadata=True,
        inference_mode="chat_with_prefix",
    )

    messages = base_model.tokenizer.messages_history[0]
    assert messages[0]["role"] == "system"
    assert "Upstream agent messages" in messages[1]["content"]
    assert "[reader | edge_weight=0.820]" in messages[1]["content"]
    assert "Focus on the egg counts." in messages[1]["content"]
    assert "[solver | edge_weight=0.810]" in messages[1]["content"]
    assert "Return the final answer" in messages[1]["content"]


def test_generate_answer_chat_mode_supports_batched_upstream_text_messages():
    base_model = _FakeBaseModel()
    agent = Agent(
        agent_id=1,
        role_config={
            "role_name": "planner",
            "system_prompt": "You plan.",
            "reasoning_steps": 4,
            "compress_last_k": 4,
        },
        base_model=base_model,
    )

    agent.generate_answer(
        task_token_ids=torch.tensor([[11, 12], [13, 14]], dtype=torch.long),
        task_attention_mask=torch.ones(2, 2, dtype=torch.long),
        upstream_prefix=None,
        upstream_text_messages=[
            [
                {"role_name": "reader", "content": "sample-0 reader", "edge_weight": 0.82},
            ],
            [
                {"role_name": "reader", "content": "sample-1 reader", "edge_weight": 0.82},
                {"role_name": "solver", "content": "sample-1 solver", "edge_weight": 0.81},
            ],
        ],
        max_new_tokens=3,
        do_sample=False,
        return_metadata=True,
        inference_mode="chat_with_prefix",
    )

    first_messages = base_model.tokenizer.messages_history[0]
    second_messages = base_model.tokenizer.messages_history[1]
    assert "sample-0 reader" in first_messages[1]["content"]
    assert "sample-1 reader" in second_messages[1]["content"]
    assert "sample-1 solver" in second_messages[1]["content"]
