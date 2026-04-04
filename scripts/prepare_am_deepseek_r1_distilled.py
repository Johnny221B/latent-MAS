#!/usr/bin/env python3
"""Prepare the AM DeepSeek R1 distilled dataset into a local JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


DATASET_REPO = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
SUBSETS = ("am_0.5M", "am_0.9M")
DEFAULT_OUTPUT = Path("data") / "am_deepseek_r1_distilled" / "train.jsonl"


def extract_message_content(messages: list[dict], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return str(message.get("content", ""))
    raise ValueError(f"Missing {role} message in sample")


def build_question_id(subset: str, user_content: str, assistant_content: str) -> str:
    stable_key = f"{subset}\n{user_content}\n{assistant_content}".encode("utf-8")
    digest = hashlib.sha1(stable_key).hexdigest()[:12]
    return f"am-r1-{digest}"


def extract_source(messages: list[dict]) -> str:
    for message in messages:
        info = message.get("info")
        if info and info.get("source"):
            return str(info["source"])
    return "unknown"


def normalize_row(row: dict, subset: str) -> dict:
    messages = row["messages"]
    user_content = extract_message_content(messages, role="user")
    assistant_content = extract_message_content(messages, role="assistant")

    return {
        "question_id": build_question_id(subset, user_content, assistant_content),
        "question": user_content,
        "answer": assistant_content,
        "subset": subset,
        "source": extract_source(messages),
    }


def download_subset_file(subset: str) -> Path:
    filename = f"{subset}.jsonl"
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
        )
    )


def iter_normalized_rows():
    for subset in SUBSETS:
        subset_path = download_subset_file(subset)
        with subset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield normalize_row(json.loads(line), subset=subset)


def write_jsonl(rows, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count % 100000 == 0:
                print(f"Wrote {count} rows to {output_path}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_jsonl(iter_normalized_rows(), args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
