import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_progress_logger_mirrors_messages_to_file_and_stdout(tmp_path: Path, capsys):
    from src.utils.progress_logging import ProgressLogger

    log_path = tmp_path / "progress.log"
    logger = ProgressLogger(log_path)

    logger.log("first line")
    logger.log("second line")

    captured = capsys.readouterr().out
    assert "first line" in captured
    assert "second line" in captured
    assert log_path.read_text(encoding="utf-8") == "first line\nsecond line\n"
