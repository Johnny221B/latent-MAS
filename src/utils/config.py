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
