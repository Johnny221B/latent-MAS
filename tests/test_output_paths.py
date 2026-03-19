from datetime import datetime
from pathlib import Path

from src.utils.output_paths import build_timestamped_output_dir


def test_build_timestamped_output_dir_appends_timestamp_without_creating_base_dir():
    result = build_timestamped_output_dir(
        "outputs/gsm8k_qwen3-8b",
        now=datetime(2026, 3, 19, 5, 30, 45),
    )

    assert result == Path("outputs/gsm8k_qwen3-8b_20260319_053045")
