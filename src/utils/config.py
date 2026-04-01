"""
Config loading and validation utilities.
"""

from pathlib import Path
import yaml


def load_config(config_path: str | Path) -> dict:
    """Load and validate an experiment config from YAML.

    Args:
        config_path: path to the YAML config file

    Returns:
        config: validated config dict
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    _normalize_config(config)
    return config


def _validate_config(config: dict):
    """Basic validation of required config fields."""
    required_sections = ["model", "graph", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    if "name" not in config["model"]:
        raise ValueError("model.name is required")
    if "config" not in config["graph"]:
        raise ValueError("graph.config is required")

    # Validate graph config file exists
    graph_path = Path(config["graph"]["config"])
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph config not found: {graph_path}")


def _normalize_config(config: dict) -> None:
    """Backfill compatible defaults and normalize deprecated aliases."""
    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("dtype", "float32")

    training_cfg = config.setdefault("training", {})
    train_strategy = training_cfg.get("train_strategy")
    if train_strategy is None:
        train_strategy = "full_finetune" if training_cfg.get("train_base_model", False) else "communication_only"
    if train_strategy not in {"communication_only", "full_finetune"}:
        raise ValueError(
            "training.train_strategy must be one of "
            "{'communication_only', 'full_finetune'}"
        )
    training_cfg["train_strategy"] = train_strategy
    training_cfg["train_base_model"] = train_strategy == "full_finetune"
    training_cfg.setdefault("save_final_checkpoint", True)
    training_cfg.setdefault("shuffle", True)
    training_cfg.setdefault("seed", 42)
    training_cfg.setdefault("resume_from_checkpoint", None)
    if isinstance(training_cfg.get("resume_from_checkpoint"), str):
        resume_path = training_cfg["resume_from_checkpoint"].strip()
        training_cfg["resume_from_checkpoint"] = resume_path or None

    evaluation_cfg = config.setdefault("evaluation", {})
    evaluation_cfg.setdefault("run_after_train", False)

    probe_cfg = config.setdefault("training_probe", {})
    probe_cfg.setdefault("enabled", False)
    probe_cfg.setdefault("samples", 100)
    probe_cfg.setdefault("seed", 42)
    probe_cfg.setdefault("every_n_steps", 0)
    probe_cfg.setdefault("batch_size", 1)
    probe_cfg.setdefault("max_new_tokens", 64)
    probe_cfg.setdefault("degenerate_max_new_tokens_ratio", 0.5)
    probe_cfg.setdefault("write_predictions_json", False)
    probe_cfg.setdefault("write_agent_logs", False)
