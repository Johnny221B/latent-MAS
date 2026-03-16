from src.models.base_model import BaseModelWrapper

model = BaseModelWrapper(
    model_name="Qwen/Qwen3-0.6B",
    cache_dir="/data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B",
)
print(f"hidden_dim: {model.hidden_dim}")
print(f"params frozen: {all(not p.requires_grad for p in model.parameters())}")