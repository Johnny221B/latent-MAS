"""AM DeepSeek R1 distilled dataset helpers and config."""

import json
from pathlib import Path


LOCAL_DATA_DIR = Path("data") / "am_deepseek_r1_distilled"


class _LocalJsonlDataset:
    """Small random-access JSONL dataset without Hugging Face cache generation."""

    def __init__(
        self,
        path: Path,
        offsets: tuple[int, ...] | None = None,
        indices: tuple[int, ...] | None = None,
    ):
        self.path = path
        self._offsets = offsets if offsets is not None else self._build_offsets(path)
        self._indices = indices
        self._file_handle = None

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict:
        resolved_idx = self._resolve_index(idx)
        offset = self._offsets[resolved_idx]
        handle = self._get_file_handle()
        handle.seek(offset)
        line = handle.readline()
        if not line:
            raise IndexError(f"Failed to read JSONL row at offset {offset}")
        return json.loads(line.decode("utf-8"))

    def select(self, indices) -> "_LocalJsonlDataset":
        selected = tuple(self._resolve_index(idx) for idx in indices)
        return _LocalJsonlDataset(self.path, offsets=self._offsets, indices=selected)

    def _resolve_index(self, idx: int) -> int:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._indices is not None:
            return self._indices[idx]
        return idx

    def _get_file_handle(self):
        if self._file_handle is None or self._file_handle.closed:
            self._file_handle = self.path.open("rb")
        return self._file_handle

    @staticmethod
    def _build_offsets(path: Path) -> tuple[int, ...]:
        offsets: list[int] = []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return tuple(offsets)

    def __del__(self):
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()


def _load_local_train_split(split: str, source: str | None = None):
    if split != "train":
        raise ValueError("am_deepseek_r1_distilled only supports the 'train' split")

    train_path = LOCAL_DATA_DIR / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            "Prepared dataset file not found: "
            f"{train_path}. Run scripts/prepare_am_deepseek_r1_distilled.py first."
        )

    # Fast path: if source is a single name and a pre-filtered file exists, use it directly
    if source is not None:
        sources = {s.strip() for s in source.split(",")}
        if len(sources) == 1:
            prefiltered = LOCAL_DATA_DIR / f"{next(iter(sources))}.jsonl"
            if prefiltered.exists():
                print(f"Using pre-filtered file: {prefiltered}")
                return _LocalJsonlDataset(prefiltered)

    dataset = _LocalJsonlDataset(train_path)

    if source is not None:
        filtered_indices = []
        for i in range(len(dataset)):
            row = dataset[i]
            if row.get("source", "unknown") in sources:
                filtered_indices.append(i)
        print(f"Filtered by source={source}: {len(filtered_indices)}/{len(dataset)} samples")
        dataset = dataset.select(filtered_indices)

    return dataset


def build_task_configs() -> dict:
    return {
        "am_deepseek_r1_distilled": {
            "loader": _load_local_train_split,
            "allowed_splits": ("train",),
            "question_id_field": "question_id",
            "question_field": "question",
            "answer_field": "answer",
            "extra_fields": ("subset", "source"),
        }
    }
