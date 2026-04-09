#!/usr/bin/env python3
"""Re-evaluate truncated samples with 2x max_new_tokens, then update eval_results.json."""
import json
import sys
import os
import time
import copy
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.cli.evaluate import (
    _build_and_load_system,
    _gpu_worker,
    _sanitize_for_gather,
    _select_batch_item,
    write_eval_snapshot,
    collate_fn,
    extract_answer,
    math_is_equivalent,
    MATH_EQUIVALENT_TASKS,
    build_agent_sample_log,
)
from src.pipeline.multi_agent_system import MultiAgentSystem
from src.utils.config import load_config
from torch.utils.data import DataLoader


def main():
    eval_json_path = sys.argv[1]  # path to eval_results.json
    token_multiplier = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    num_gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print(f"Loading {eval_json_path}")
    with open(eval_json_path) as f:
        data = json.load(f)

    params = data["parameters"]
    config = params["config"]
    checkpoint_path = params["checkpoint_path"]
    task = data["task"]
    old_max_tokens = params["generation_max_new_tokens"]
    new_max_tokens = old_max_tokens * token_multiplier
    inference_mode = params["inference_mode"]
    use_terminal_prefix = params["use_terminal_prefix"]
    do_sample = params["do_sample"]
    batch_size = params.get("batch_size", 1)

    # Find truncated samples
    truncated_samples = [s for s in data["samples"] if s.get("truncated")]
    if not truncated_samples:
        print("No truncated samples found.")
        return

    print(f"Found {len(truncated_samples)} truncated samples")
    print(f"Re-evaluating with max_new_tokens: {old_max_tokens} -> {new_max_tokens}")

    # Build dataset from truncated samples
    dataset_items = []
    for s in truncated_samples:
        dataset_items.append({
            "question_id": s["question_id"],
            "question": s["question"],
            "answer": s["gold"],
        })

    # Use multiprocessing multi-GPU eval
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_gpus_use = min(num_gpus, num_gpus_available, len(dataset_items))

    # Split dataset into per-GPU shards
    shards = []
    for i in range(num_gpus_use):
        shard = [dataset_items[j] for j in range(i, len(dataset_items), num_gpus_use)]
        shards.append(shard)

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    print(f"Spawning {num_gpus_use} GPU workers...")
    workers = []
    for gpu_id in range(num_gpus_use):
        if not shards[gpu_id]:
            continue
        p = mp.Process(
            target=_gpu_worker,
            args=(
                gpu_id, shards[gpu_id], config, checkpoint_path, task,
                new_max_tokens, inference_mode, use_terminal_prefix,
                do_sample, False, batch_size, result_queue,
            ),
        )
        p.start()
        workers.append(p)

    active_workers = len(workers)
    new_results = {}
    progress = tqdm(total=len(dataset_items), desc="Re-eval truncated")
    done_count = 0
    while done_count < active_workers:
        item = result_queue.get()
        if item is None:
            done_count += 1
            continue
        qid = item["question_id"]
        new_results[qid] = {
            "question": item["question"],
            "question_id": qid,
            "gold": item["gold"],
            "prediction": item["prediction"],
            "generation": item["generation"],
            "correct": item["correct"],
            "truncated": item.get("truncated", False),
        }
        status = "correct" if item["correct"] else "wrong"
        tc = item["generation"].get("generated_token_count", 0)
        still_trunc = " [STILL TRUNCATED]" if item.get("truncated") else ""
        print(f"  {qid}: {status}, tokens={tc}{still_trunc}")
        progress.update(1)

    progress.close()
    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    # Update original results
    updated = 0
    for i, s in enumerate(data["samples"]):
        if s["question_id"] in new_results:
            data["samples"][i] = new_results[s["question_id"]]
            updated += 1

    # Recompute metrics
    correct = sum(1 for s in data["samples"] if s["correct"])
    total = len(data["samples"])
    valid = sum(1 for s in data["samples"] if not s.get("truncated"))
    valid_correct = sum(1 for s in data["samples"] if s["correct"] and not s.get("truncated"))
    data["metrics"]["correct"] = correct
    data["metrics"]["total"] = total
    data["metrics"]["accuracy"] = correct / total * 100 if total > 0 else 0.0
    data["metrics"]["valid"] = valid
    data["metrics"]["valid_correct"] = valid_correct
    data["metrics"]["valid_accuracy"] = valid_correct / valid * 100 if valid > 0 else 0.0
    data["metrics"]["truncated"] = total - valid

    with open(eval_json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated {updated} samples in {eval_json_path}")
    print(f"New accuracy: {data['metrics']['accuracy']:.1f}% ({correct}/{total})")
    print(f"Still truncated: {total - valid}")


if __name__ == "__main__":
    main()
