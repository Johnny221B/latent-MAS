import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_model import BaseModelWrapper
from src.graph.dag_executor import DAGExecutor
from src.models.agent import Agent
from src.models.compressor import PrefixProjector
from src.pipeline.multi_agent_system import _get_kv_head_dim


def _make_dynamic_cache(
    batch_size: int = 1,
    num_layers: int = 2,
    num_kv_heads: int = 1,
    seq_len: int = 3,
    head_dim: int = 2,
) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        key_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        value_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        cache.update(key_states, value_states, layer_idx)
    return cache


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(
        self,
        prompts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=False,
        **kwargs,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        input_ids = torch.tensor([[5, 6], [7, 8]], dtype=torch.long)[:batch_size]
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [f"question-{idx}" for idx in range(token_ids.shape[0])]

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ):
        return messages[-1]["content"]

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "|".join(str(x) for x in token_ids)


class _RecordingBaseModel:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.device = torch.device("cpu")
        self.model_config = SimpleNamespace(
            hidden_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        self.forward_kwargs = None
        self.reason_kwargs = None

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        prefix_embeds=None,
        past_key_values=None,
        output_hidden_states=True,
    ):
        self.forward_kwargs = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": None if attention_mask is None else attention_mask.detach().clone(),
            "prefix_embeds": prefix_embeds,
            "past_key_values": past_key_values,
            "output_hidden_states": output_hidden_states,
        }
        batch_size, seq_len = input_ids.shape
        vocab_size = 8
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        return {
            "logits": logits,
            "last_hidden_state": torch.zeros(batch_size, seq_len, self.model_config.hidden_size),
            "full_last_hidden_state": torch.zeros(batch_size, seq_len, self.model_config.hidden_size),
            "prefix_len": 0,
        }

    def latent_reasoning(
        self,
        input_ids,
        attention_mask=None,
        prefix_embeds=None,
        past_key_values=None,
        num_latent_steps=1,
    ):
        self.reason_kwargs = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": None if attention_mask is None else attention_mask.detach().clone(),
            "prefix_embeds": prefix_embeds,
            "past_key_values": past_key_values,
            "num_latent_steps": num_latent_steps,
        }
        batch_size = input_ids.shape[0]
        hidden_dim = self.model_config.hidden_size
        return {
            "hidden_trajectory": torch.zeros(batch_size, num_latent_steps, hidden_dim),
            "prefix_len": 0 if past_key_values is None else past_key_values.get_seq_length(),
        }


def test_prefix_projector_builds_per_layer_dynamic_cache():
    projector = PrefixProjector(
        num_layers=3,
        hidden_dim=8,
        num_kv_heads=2,
        head_dim=4,
    )
    prefix_embeds = torch.randn(2, 5, 8, requires_grad=True)

    cache = projector(prefix_embeds)

    assert isinstance(cache, DynamicCache)
    assert len(cache.layers) == 3
    assert cache.get_seq_length() == 5
    assert cache.layers[0].keys.shape == (2, 2, 5, 4)
    assert cache.layers[0].values.shape == (2, 2, 5, 4)

    loss = sum(layer.keys.sum() + layer.values.sum() for layer in cache.layers)
    loss.backward()

    assert projector.mlp[0].weight.grad is not None
    assert projector.mlp[2].weight.grad is not None


def test_get_kv_head_dim_prefers_explicit_config_value():
    cfg = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=32,
        head_dim=128,
    )

    assert _get_kv_head_dim(cfg) == 128


def test_get_kv_head_dim_falls_back_to_attention_ratio():
    cfg = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=32,
    )

    assert _get_kv_head_dim(cfg) == 80


def test_base_model_wrapper_forwards_past_key_values_without_embedding_prefix_concat():
    class FakeHFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 4)
            self.last_kwargs = None

        def get_input_embeddings(self):
            return self.embedding

        def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            output_hidden_states=True,
            return_dict=False,
            **kwargs,
        ):
            self.last_kwargs = {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
            if inputs_embeds is None:
                inputs_embeds = self.embedding(input_ids)
            batch_size, seq_len, _ = inputs_embeds.shape
            logits = torch.zeros(batch_size, seq_len, 16)
            hidden_states = (inputs_embeds, inputs_embeds)
            return type(
                "FakeOutput",
                (),
                {
                    "logits": logits,
                    "past_key_values": past_key_values,
                    "hidden_states": hidden_states,
                },
            )()

    wrapper = BaseModelWrapper.__new__(BaseModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = FakeHFModel()
    wrapper.model_config = SimpleNamespace(hidden_size=4)

    cache = _make_dynamic_cache(seq_len=3)
    output = wrapper(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        past_key_values=cache,
        output_hidden_states=True,
    )

    assert output["logits"].shape == (1, 2, 16)
    assert wrapper.model.last_kwargs["input_ids"].shape == (1, 2)
    assert wrapper.model.last_kwargs["inputs_embeds"] is None
    assert wrapper.model.last_kwargs["past_key_values"] is cache


def test_agent_reason_uses_past_key_values_for_latent_reasoning():
    base_model = _RecordingBaseModel()
    agent = Agent(
        agent_id=0,
        role_config={
            "role_name": "planner",
            "system_prompt": "Plan carefully.",
            "reasoning_steps": 3,
            "compress_last_k": 3,
        },
        base_model=base_model,
    )
    upstream_prefix_kv = _make_dynamic_cache()

    output = agent.reason(
        task_token_ids=torch.tensor([[11, 12]], dtype=torch.long),
        task_attention_mask=torch.ones(1, 2, dtype=torch.long),
        upstream_prefix_kv=upstream_prefix_kv,
        prefix_len=3,
    )

    assert output["hidden_trajectory"].shape == (1, 3, 4)
    assert base_model.reason_kwargs["past_key_values"] is upstream_prefix_kv
    assert base_model.reason_kwargs["prefix_embeds"] is None


def test_agent_forward_for_loss_passes_kv_cache_and_expands_attention_mask():
    base_model = _RecordingBaseModel()
    agent = Agent(
        agent_id=1,
        role_config={
            "role_name": "solver",
            "system_prompt": "Solve carefully.",
            "reasoning_steps": 2,
            "compress_last_k": 2,
        },
        base_model=base_model,
    )
    upstream_prefix_kv = _make_dynamic_cache(seq_len=4)

    result = agent.forward_for_loss(
        task_token_ids=torch.tensor([[3, 4]], dtype=torch.long),
        task_attention_mask=torch.ones(1, 2, dtype=torch.long),
        upstream_prefix_kv=upstream_prefix_kv,
        prefix_len=4,
        answer_ids=torch.tensor([[5, 6]], dtype=torch.long),
        answer_mask=torch.ones(1, 2, dtype=torch.long),
        input_mode="chat_with_prefix",
    )

    assert result["answer_len"] == 2
    assert base_model.forward_kwargs["past_key_values"] is upstream_prefix_kv
    assert base_model.forward_kwargs["prefix_embeds"] is None
    assert base_model.forward_kwargs["attention_mask"].shape[1] == 8


def test_dag_executor_projects_prefix_before_reason_and_terminal_generation():
    class FakeAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.role_name = f"agent-{agent_id}"
            self.system_prompt = f"prompt-{agent_id}"
            self.reason_kwargs = None
            self.generate_kwargs = None

        def reason(self, **kwargs):
            self.reason_kwargs = kwargs
            return {
                "hidden_trajectory": torch.ones(1, 2, 4),
                "compressor_mask": torch.ones(1, 2),
                "prefix_len": kwargs.get("prefix_len", 0),
            }

        def generate_answer(self, **kwargs):
            self.generate_kwargs = kwargs
            return {
                "generated_text": "42",
                "finish_reason": "eos",
                "generated_token_count": 1,
                "stopped_early": True,
            }

    class FakeCompressor:
        def __call__(self, hidden, mask=None):
            return hidden[:, :1, :]

    class RecordingPrefixProjector:
        def __init__(self):
            self.calls = []

        def __call__(self, prefix_embeds):
            self.calls.append(prefix_embeds.detach().clone())
            return _make_dynamic_cache(seq_len=prefix_embeds.shape[1])

    agents = [FakeAgent(0), FakeAgent(1)]
    projector = RecordingPrefixProjector()
    executor = DAGExecutor()

    out = executor.execute(
        agents=agents,
        adjacency=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        compressor=FakeCompressor(),
        prefix_projector=projector,
        task_token_ids=torch.tensor([[1, 2]], dtype=torch.long),
        task_attention_mask=torch.ones(1, 2, dtype=torch.long),
        training=False,
        inference_mode="chat_with_prefix",
    )

    assert out["generated_text"] == "42"
    assert len(projector.calls) == 1
    assert agents[0].reason_kwargs["upstream_prefix_kv"] is None
    assert agents[1].generate_kwargs["upstream_prefix_kv"] is not None
    assert agents[1].generate_kwargs["prefix_len"] == 1


def test_agent_generate_answer_accepts_kv_prefix_on_tiny_real_qwen():
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B", local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Local Qwen3 config unavailable: {exc}")

    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    cfg.vocab_size = 128
    cfg.tie_word_embeddings = True
    cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    cfg.max_window_layers = cfg.num_hidden_layers
    cfg.sliding_window = None
    cfg.use_sliding_window = False
    cfg.dtype = "float32"

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def batch_decode(self, token_ids, skip_special_tokens=True):
            return [f"q-{i}" for i in range(token_ids.shape[0])]

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        ):
            return messages[-1]["content"]

        def __call__(
            self,
            prompts,
            return_tensors=None,
            padding=True,
            truncation=True,
            max_length=None,
            add_special_tokens=False,
            **kwargs,
        ):
            if isinstance(prompts, str):
                prompts = [prompts]
            batch = len(prompts)
            input_ids = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long)[:batch]
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def decode(self, token_ids, skip_special_tokens=True):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return " ".join(str(x) for x in token_ids)

    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    model.eval()

    wrapper = BaseModelWrapper.__new__(BaseModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.model_config = model.config
    wrapper.tokenizer = TinyTokenizer()
    wrapper.base_model_trainable = False

    agent = Agent(
        agent_id=0,
        role_config={
            "role_name": "solver",
            "system_prompt": "Solve carefully.",
            "reasoning_steps": 2,
            "compress_last_k": 2,
        },
        base_model=wrapper,
    )

    projector = PrefixProjector(
        num_layers=cfg.num_hidden_layers,
        hidden_dim=cfg.hidden_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        cache_config=cfg,
    )
    past = projector(torch.randn(1, 3, cfg.hidden_size))

    result = agent.generate_answer(
        task_token_ids=torch.tensor([[11, 12]], dtype=torch.long),
        task_attention_mask=torch.ones(1, 2, dtype=torch.long),
        upstream_prefix_kv=past,
        prefix_len=3,
        max_new_tokens=2,
        do_sample=False,
        return_metadata=True,
        inference_mode="chat_with_prefix",
    )

    assert result["used_upstream_prefix"] is True
    assert isinstance(result["generated_text"], str)
