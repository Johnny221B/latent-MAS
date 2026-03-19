import copy
import os
from pathlib import Path


def load_env_file(env_path: str | Path) -> dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def resolve_env_value(key: str, env_path: str | Path | None = None) -> str | None:
    current = os.environ.get(key)
    if current:
        return current

    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    env_values = load_env_file(env_path)
    value = env_values.get(key)
    if value:
        os.environ[key] = value
    return value


def get_report_config(config: dict) -> dict:
    return copy.deepcopy(config.get("report", {}))


def should_enable_wandb(config: dict) -> bool:
    report_cfg = get_report_config(config)
    return bool(report_cfg.get("use_wandb", False))


def init_wandb_run(config: dict, output_dir: str | Path, rank: int = 0):
    if rank != 0 or not should_enable_wandb(config):
        return None

    report_cfg = get_report_config(config)
    key_env = report_cfg.get("key_env", "WANDB_API_KEY")
    env_file = report_cfg.get("env_file")
    api_key = resolve_env_value(key_env, env_file)
    if not api_key:
        print(f"W&B disabled: {key_env} not found in environment or .env")
        return None

    try:
        import wandb
    except ImportError:
        print("W&B disabled: package 'wandb' is not installed")
        return None

    os.environ["WANDB_API_KEY"] = api_key

    output_dir = Path(output_dir)
    run = wandb.init(
        project=report_cfg["project"],
        entity=report_cfg.get("entity"),
        name=report_cfg.get("run_name", output_dir.name),
        dir=str(output_dir),
        config=copy.deepcopy(config),
        mode=report_cfg.get("mode", "online"),
        tags=report_cfg.get("tags"),
        notes=report_cfg.get("notes"),
    )
    return run


def log_wandb(run, payload: dict, step: int | None = None):
    if run is None:
        return
    run.log(payload, step=step)


def finish_wandb(run):
    if run is None:
        return
    run.finish()
