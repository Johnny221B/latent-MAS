from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_timestamped_output_dir(base_output_dir: str | Path, now: datetime | None = None) -> Path:
    base_path = Path(base_output_dir)
    current_time = now or datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    return Path(f"{base_path}_{timestamp}")
