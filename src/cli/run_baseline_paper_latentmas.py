import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch.distributed as dist


def extract_last_json_object(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("No JSON result found in LatentMAS output")


def normalize_result(result: dict) -> dict:
    normalized = dict(result)
    accuracy = normalized.get("accuracy")
    if isinstance(accuracy, (int, float)):
        normalized["accuracy_raw"] = float(accuracy)
        normalized["accuracy"] = float(accuracy) * 100 if accuracy <= 1.0 else float(accuracy)
    return normalized


def setup_distributed() -> tuple[int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="gloo")
        return rank, world_size, True
    return 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_sharded_objects(local_obj, world_size: int):
    if world_size <= 1:
        return [local_obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_obj)
    return gathered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--latent-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rank, world_size, is_dist = setup_distributed()

    root = Path(__file__).resolve().parents[2]
    latentmas_dir = root / "LatentMAS"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model_name.split("/")[-1].lower()
    sample_tag = str(args.max_samples)
    output_path = Path(args.output) if args.output else output_dir / f"paper_latentmas_{slug}_{sample_tag}.json"
    shard_output_path = output_dir / f".paper_latentmas_{slug}_{sample_tag}_rank{rank}.json"

    cmd = [
        sys.executable,
        "run.py",
        "--method",
        "latent_mas",
        "--model_name",
        args.model_name,
        "--task",
        "gsm8k",
        "--prompt",
        args.prompt,
        "--max_samples",
        str(args.max_samples),
        "--latent_steps",
        str(args.latent_steps),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--shard_rank",
        str(rank),
        "--shard_world_size",
        str(world_size),
        "--output_json",
        str(shard_output_path),
    ]

    started_at = time.time()
    proc = subprocess.run(
        cmd,
        cwd=latentmas_dir,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    payload = json.loads(shard_output_path.read_text())
    raw_result = normalize_result(payload["summary"])
    local_result = {
        "method": "paper_latentmas",
        "metrics": {
            "accuracy": raw_result.get("accuracy"),
            "accuracy_raw": raw_result.get("accuracy_raw"),
            "correct": raw_result.get("correct"),
            "total": raw_result.get("max_samples"),
            "total_time_sec": raw_result.get("total_time_sec"),
            "time_per_sample_sec": raw_result.get("time_per_sample_sec"),
        },
        "parameters": {
            "model_name": args.model_name,
            "task": "gsm8k",
            "max_samples": args.max_samples,
            "prompt": args.prompt,
            "latent_steps": args.latent_steps,
            "max_new_tokens": args.max_new_tokens,
            "wrapped_command": cmd,
            "wrapper_elapsed_sec": round(time.time() - started_at, 4),
            "world_size": world_size,
        },
        "raw_result": raw_result,
        "samples": payload.get("preds", []),
    }
    gathered = gather_sharded_objects(local_result, world_size)
    if rank != 0:
        if is_dist:
            dist.barrier()
            cleanup_distributed()
        return

    merged_samples = []
    total = 0
    correct = 0
    total_time = 0.0
    wrapper_elapsed = 0.0
    for shard in gathered:
        total += shard["metrics"]["total"] or 0
        correct += shard["metrics"]["correct"] or 0
        total_time += shard["metrics"]["total_time_sec"] or 0.0
        wrapper_elapsed = max(wrapper_elapsed, shard["parameters"]["wrapper_elapsed_sec"])
        merged_samples.extend(shard.get("samples", []))
    result = {
        "method": "paper_latentmas",
        "metrics": {
            "accuracy": (correct / total * 100) if total > 0 else 0.0,
            "accuracy_raw": (correct / total) if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "total_time_sec": total_time,
            "time_per_sample_sec": (total_time / total) if total > 0 else None,
        },
        "parameters": {
            "model_name": args.model_name,
            "task": "gsm8k",
            "max_samples": args.max_samples,
            "prompt": args.prompt,
            "latent_steps": args.latent_steps,
            "max_new_tokens": args.max_new_tokens,
            "wrapped_command": cmd,
            "wrapper_elapsed_sec": round(wrapper_elapsed, 4),
            "world_size": world_size,
        },
        "samples": merged_samples,
        "shards": gathered,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")
    if is_dist:
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
