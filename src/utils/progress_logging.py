from __future__ import annotations

from pathlib import Path


class ProgressLogger:
    """Mirror selected progress messages to stdout and a local log file."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str = "") -> None:
        print(message)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message)
            handle.write("\n")

