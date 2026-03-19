"""
Evaluation script: test trained model on GSM8K test set.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/cli/evaluate.py \
        --config outputs/gsm8k_qwen3-0.6b_20260317_225851/config.yaml \
        --checkpoint outputs/gsm8k_qwen3-0.6b_20260317_225851/final_model.pt \
        --max_samples 100
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.utils.config import load_config
from src.utils.answer_extraction import extract_answer
from src.pipeline.multi_agent_system import MultiAgentSystem
from data.dataset import create_dataset
from src.models.agent import Agent


def append_jsonl_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_partial_eval_snapshot(
    partial_path: Path,
    method: str,
    task: str,
    correct: int,
    total: int,
    time_seconds: float,
    parameters: dict,
    world_size: int,
    jsonl_path: Path,
) -> None:
    accuracy = correct / total * 100 if total > 0 else 0.0
    payload = {
        "method": method,
        "task": task,
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "time_seconds": time_seconds,
        },
        "parameters": parameters,
        "world_size": world_size,
        "samples_jsonl": str(jsonl_path),
    }
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def build_generation_metadata(generated_token_ids: list[int], eos_token_id: int | None, max_new_tokens: int) -> dict:
    finish_reason = Agent._infer_finish_reason(
        generated_ids=generated_token_ids,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    return {
        "finish_reason": finish_reason,
        "generated_token_count": len(generated_token_ids),
        "stopped_early": finish_reason != "max_new_tokens",
    }


def setup_eval_distributed() -> tuple[torch.device, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return device, rank, world_size, True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, 1, False


def cleanup_eval_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def shard_dataset(dataset, rank: int, world_size: int):
    if world_size <= 1:
        return dataset
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)


def gather_sharded_objects(local_obj, rank: int, world_size: int):
    if world_size <= 1:
        return [local_obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_obj)
    return gathered


def evaluate(
    config_path: str,
    checkpoint_path: str,
    max_samples: int | None = None,
    max_new_tokens: int = 2048,
):
    config = load_config(config_path)
    generation_max_new_tokens = max_new_tokens
    device, rank, world_size, is_dist = setup_eval_distributed()
    if is_main_process(rank):
        print(f"Device: {device}")
        print(f"World size: {world_size}")

    # ── Build system ──
    if is_main_process(rank):
        print("Building multi-agent system...")
    system = MultiAgentSystem(config)

    # ── Load checkpoint ──
    if is_main_process(rank):
        print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Handle DDP-wrapped state dict (keys may have "module." prefix)
    comp_state = ckpt["compressor_state"]
    cleaned_state = {}
    for k, v in comp_state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state[new_key] = v
    system.compressor.load_state_dict(cleaned_state)

    system.adjacency.load_state_dict(ckpt["adjacency_state"])
    system.to(device)
    system.eval()

    # ── Show learned adjacency ──
    if is_main_process(rank):
        print(f"\nLearned adjacency:")
        print(system.log_adjacency())
        A = system.adjacency.get_adjacency().detach()
        print(f"Range: [{A.min().item():.4f}, {A.max().item():.4f}]")
        print(f"Hard adjacency (threshold=0.5):")
        print(system.adjacency.get_hard_adjacency())

    # ── Dataset ──
    task = config["training"]["task"]
    if max_samples is not None and max_samples < 0:
        max_samples = None
    if is_main_process(rank):
        print(f"\nLoading {task} test set...")
    full_dataset = create_dataset(task=task, split="test", max_samples=max_samples)
    dataset = shard_dataset(full_dataset, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    if is_main_process(rank):
        print(f"Test samples: {len(full_dataset)}")
        if world_size > 1:
            print(f"Sharded samples per rank: {[len(range(r, len(full_dataset), world_size)) for r in range(world_size)]}")

    # ── Evaluate ──
    correct = 0
    total = 0
    results = []
    checkpoint_dir = Path(checkpoint_path).parent
    jsonl_path = checkpoint_dir / f"eval_samples.rank{rank}.jsonl"
    partial_eval_path = checkpoint_dir / f"eval_results.partial.rank{rank}.json"
    partial_baseline_path = checkpoint_dir / f"eval_results.baseline_single.partial.rank{rank}.json"
    for path in (jsonl_path, partial_eval_path, partial_baseline_path):
        if path.exists():
            path.unlink()

    if is_main_process(rank):
        print("\nRunning evaluation...")
    t_start = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            t0 = time.time()

            # Tokenize question
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=config["training"].get("max_seq_len", 2048),
            )
            task_ids = tokenized["input_ids"].to(device)
            task_mask = tokenized["attention_mask"].to(device)

            # Run inference (no answer_ids → triggers generate_answer path)
            output = system(
                task_token_ids=task_ids,
                task_attention_mask=task_mask,
            )

            generated_text = output["generated_text"]
            generation = output.get("generation", {})

            # Extract and compare answer
            pred = extract_answer(generated_text, task_type=task)
            gold = batch["answers"][0].strip()
            is_correct = pred.strip() == gold.strip()

            if is_correct:
                correct += 1
            total += 1

            t1 = time.time()

            results.append({
                "question": batch["questions"][0],
                "gold": gold,
                "prediction": pred,
                "generated_text": generated_text[:500],
                "generation": generation,
                "correct": is_correct,
            })
            append_jsonl_record(
                jsonl_path,
                {
                    "phase": "ours",
                    "rank": rank,
                    "question": batch["questions"][0],
                    "gold": gold,
                    "prediction": pred,
                    "generated_text": generated_text[:500],
                    "generation": generation,
                    "correct": is_correct,
                },
            )
            write_partial_eval_snapshot(
                partial_path=partial_eval_path,
                method="ours_trained_multi_agent",
                task=task,
                correct=correct,
                total=total,
                time_seconds=time.time() - t_start,
                parameters={
                    "config_path": config_path,
                    "checkpoint_path": checkpoint_path,
                    "max_samples": max_samples,
                    "generation_max_new_tokens": generation_max_new_tokens,
                    "config": copy.deepcopy(config),
                    "rank": rank,
                },
                world_size=world_size,
                jsonl_path=jsonl_path,
            )

            # Print progress
            if is_main_process(rank) and ((idx + 1) % 10 == 0 or (idx + 1) == len(dataloader)):
                acc = correct / total * 100
                print(
                    f"  [rank{rank} {idx+1}/{len(dataloader)}] "
                    f"Acc: {acc:.1f}% ({correct}/{total}) | "
                    f"{t1-t0:.1f}s/sample"
                )

            # Print some examples
            if is_main_process(rank) and idx < 5:
                status = "✓" if is_correct else "✗"
                print(f"\n  Example {idx+1} [{status}]:")
                print(f"    Q: {batch['questions'][0][:100]}...")
                print(f"    Gold: {gold}")
                print(f"    Pred: {pred}")
                print(f"    Gen:  {generated_text[:200]}")

    local_eval = {
        "correct": correct,
        "total": total,
        "samples": results,
        "time_seconds": time.time() - t_start,
    }
    gathered_eval = gather_sharded_objects(local_eval, rank, world_size)

    if not is_main_process(rank):
        if is_dist:
            dist.barrier()
            cleanup_eval_distributed()
        return

    # ── Summary ──
    all_results = []
    correct = 0
    total = 0
    t_total = 0.0
    for shard in gathered_eval:
        correct += shard["correct"]
        total += shard["total"]
        all_results.extend(shard["samples"])
        t_total = max(t_total, shard["time_seconds"])
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Task:     {task}")
    print(f"  Samples:  {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Time:     {t_total:.1f}s ({t_total/total:.1f}s/sample)")
    print(f"{'='*60}")

    # ── Save results ──
    eval_path = checkpoint_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump({
            "method": "ours_trained_multi_agent",
            "task": task,
            "metrics": {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "time_seconds": t_total,
            },
            "parameters": {
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "max_samples": max_samples,
                "generation_max_new_tokens": generation_max_new_tokens,
                "config": copy.deepcopy(config),
            },
            "world_size": world_size,
            "samples": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {eval_path}")

    # ── Also run baseline (no multi-agent, just the model) ──
    print(f"\n{'='*60}")
    print(f"  BASELINE (single model, no multi-agent)")
    print(f"{'='*60}")

    baseline_correct = 0
    baseline_total = 0
    baseline_samples = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=config["training"].get("max_seq_len", 2048),
            )
            task_ids = tokenized["input_ids"].to(device)
            task_mask = tokenized["attention_mask"].to(device)

            # Direct generation without any prefix
            gen_out = system.base_model.model.generate(
                input_ids=task_ids,
                attention_mask=task_mask,
                max_new_tokens=generation_max_new_tokens,
                do_sample=False,
                pad_token_id=system.base_model.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out[0]
            generated_token_ids = sequences[0][task_ids.shape[1]:].tolist()
            baseline_text = system.base_model.tokenizer.decode(
                generated_token_ids, skip_special_tokens=True,
            )
            generation = build_generation_metadata(
                generated_token_ids=generated_token_ids,
                eos_token_id=system.base_model.tokenizer.eos_token_id,
                max_new_tokens=generation_max_new_tokens,
            )

            pred = extract_answer(baseline_text, task_type=task)
            gold = batch["answers"][0].strip()
            is_correct = pred.strip() == gold.strip()
            if is_correct:
                baseline_correct += 1
            baseline_total += 1
            baseline_samples.append({
                "question": batch["questions"][0],
                "gold": gold,
                "prediction": pred,
                "generated_text": baseline_text[:500],
                "generation": generation,
                "correct": is_correct,
            })
            append_jsonl_record(
                jsonl_path,
                {
                    "phase": "baseline_single_model",
                    "rank": rank,
                    "question": batch["questions"][0],
                    "gold": gold,
                    "prediction": pred,
                    "generated_text": baseline_text[:500],
                    "generation": generation,
                    "correct": is_correct,
                },
            )
            write_partial_eval_snapshot(
                partial_path=partial_baseline_path,
                method="single_model",
                task=task,
                correct=baseline_correct,
                total=baseline_total,
                time_seconds=time.time() - t_start,
                parameters={
                    "generation_max_new_tokens": generation_max_new_tokens,
                    "do_sample": False,
                    "world_size": world_size,
                    "rank": rank,
                },
                world_size=world_size,
                jsonl_path=jsonl_path,
            )

            if is_main_process(rank) and (idx + 1) % 50 == 0:
                bacc = baseline_correct / baseline_total * 100
                print(f"  [{idx+1}/{len(dataloader)}] Baseline Acc: {bacc:.1f}%")

    local_baseline = {
        "correct": baseline_correct,
        "total": baseline_total,
        "samples": baseline_samples,
    }
    gathered_baseline = gather_sharded_objects(local_baseline, rank, world_size)

    baseline_samples = []
    baseline_correct = 0
    baseline_total = 0
    for shard in gathered_baseline:
        baseline_correct += shard["correct"]
        baseline_total += shard["total"]
        baseline_samples.extend(shard["samples"])
    baseline_acc = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Multi-agent (trained): {accuracy:.2f}%")
    print(f"  Single model baseline: {baseline_acc:.2f}%")
    print(f"  Improvement:           {accuracy - baseline_acc:+.2f}%")
    print(f"{'='*60}")

    # Append baseline to results
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    eval_data["baseline_single_model"] = {
        "method": "single_model",
        "metrics": {
            "accuracy": baseline_acc,
            "correct": baseline_correct,
            "total": baseline_total,
        },
        "parameters": {
            "generation_max_new_tokens": generation_max_new_tokens,
            "do_sample": False,
            "world_size": world_size,
        },
        "samples": baseline_samples,
    }
    eval_data["comparison"] = {
        "baseline_accuracy": baseline_acc,
        "improvement": accuracy - baseline_acc,
    }
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    if is_dist:
        dist.barrier()
    cleanup_eval_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.max_samples, args.max_new_tokens)
