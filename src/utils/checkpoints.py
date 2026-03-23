from pathlib import Path


def find_latest_complete_checkpoint_dir(outputs_dir: str | Path) -> Path:
    """Return the newest run directory containing both config and final weights."""
    outputs_dir = Path(outputs_dir)
    candidates = [
        path
        for path in outputs_dir.iterdir()
        if path.is_dir()
        and (path / "config.yaml").is_file()
        and (path / "final_model.pt").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete checkpoint directories found under {outputs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)
