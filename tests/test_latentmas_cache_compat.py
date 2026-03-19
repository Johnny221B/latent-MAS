import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "LatentMAS"))

from models import _past_length


class FakeDynamicCache:
    def get_seq_length(self):
        return 123


def test_past_length_supports_dynamic_cache():
    assert _past_length(FakeDynamicCache()) == 123
