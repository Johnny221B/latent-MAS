import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_find_latest_complete_checkpoint_dir_prefers_newest_complete_run(tmp_path: Path):
    from src.utils.checkpoints import find_latest_complete_checkpoint_dir

    older = tmp_path / "gsm8k_old"
    older.mkdir()
    (older / "config.yaml").write_text("model: {}\ngraph: {config: graph.json}\ntraining: {task: gsm8k}\n", encoding="utf-8")
    (older / "final_model.pt").write_bytes(b"old")

    incomplete = tmp_path / "gsm8k_incomplete"
    incomplete.mkdir()
    (incomplete / "config.yaml").write_text("model: {}\ngraph: {config: graph.json}\ntraining: {task: gsm8k}\n", encoding="utf-8")

    newest = tmp_path / "gsm8k_new"
    newest.mkdir()
    (newest / "config.yaml").write_text("model: {}\ngraph: {config: graph.json}\ntraining: {task: gsm8k}\n", encoding="utf-8")
    (newest / "final_model.pt").write_bytes(b"new")

    os.utime(older, (100, 100))
    os.utime(incomplete, (300, 300))
    os.utime(newest, (200, 200))

    resolved = find_latest_complete_checkpoint_dir(tmp_path)

    assert resolved == newest


def test_find_latest_complete_checkpoint_dir_raises_when_no_complete_run(tmp_path: Path):
    from src.utils.checkpoints import find_latest_complete_checkpoint_dir

    incomplete = tmp_path / "gsm8k_incomplete"
    incomplete.mkdir()
    (incomplete / "config.yaml").write_text("model: {}\ngraph: {config: graph.json}\ntraining: {task: gsm8k}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No complete checkpoint directories found"):
        find_latest_complete_checkpoint_dir(tmp_path)
