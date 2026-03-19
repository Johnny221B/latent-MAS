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


def build_agent_sample_log(
    question: str,
    gold: str,
    prediction: str,
    generated_text: str,
    generation: dict,
    correct: bool,
    agent_logs: list[dict],
) -> dict:
    return {
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "generated_text": generated_text[:500],
        "generation": generation,
        "correct": correct,
        "agents": agent_logs,
    }


def write_eval_snapshot(
    eval_path: Path,
    method: str,
    task: str,
    correct: int,
    total: int,
    time_seconds: float,
    parameters: dict,
    world_size: int,
    samples: list[dict],
    baseline_single_model: dict | None = None,
    comparison: dict | None = None,
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
        "samples": samples,
    }
    if baseline_single_model is not None:
        payload["baseline_single_model"] = baseline_single_model
    if comparison is not None:
        payload["comparison"] = comparison
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_agent_log_snapshot(
    agent_log_path: Path,
    method: str,
    task: str,
    parameters: dict,
    samples: list[dict],
) -> None:
    payload = {
        "method": method,
        "task": task,
        "parameters": parameters,
        "samples": samples,
    }
    agent_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agent_log_path, "w", encoding="utf-8") as f:
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
            device_count = max(torch.cuda.device_count(), 1)
            device_index = local_rank % device_count
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
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
    inference_mode: str = "chat_with_prefix",
    use_terminal_prefix: bool = True,
    run_baseline: bool = False,
    do_sample: bool = False,
    write_agent_logs: bool = True,
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
    agent_log_results = []
    checkpoint_dir = Path(checkpoint_path).parent
    eval_path = checkpoint_dir / "eval_results.json"
    agent_log_path = checkpoint_dir / "agent_logs.json"
    if is_main_process(rank):
        cleanup_patterns = [
            "eval_results.partial.rank*.json",
            "eval_results.rank*.json",
            "eval_samples.rank*.jsonl",
            "agent_logs.rank*.jsonl",
        ]
        for pattern in cleanup_patterns:
            for path in checkpoint_dir.glob(pattern):
                path.unlink()
        for path in (eval_path, agent_log_path):
            if path.exists():
                path.unlink()

    if is_main_process(rank):
        print("\nRunning evaluation...")
    t_start = time.time()

    local_steps = len(dataloader)
    step_counts = gather_sharded_objects(local_steps, rank, world_size)
    max_steps = max(step_counts)
    dataloader_iter = iter(dataloader)

    with torch.no_grad():
        for idx in range(max_steps):
            batch = None
            local_update = None
            t0 = time.time()

            if idx < local_steps:
                batch = next(dataloader_iter)

                tokenized = system.base_model.tokenize(
                    batch["questions"],
                    max_length=config["training"].get("max_seq_len", 2048),
                )
                task_ids = tokenized["input_ids"].to(device)
                task_mask = tokenized["attention_mask"].to(device)

                output = system(
                    task_token_ids=task_ids,
                    task_attention_mask=task_mask,
                    max_new_tokens=generation_max_new_tokens,
                    inference_mode=inference_mode,
                    use_terminal_prefix=use_terminal_prefix,
                    do_sample=do_sample,
                    collect_agent_logs=write_agent_logs,
                )

                generated_text = output["generated_text"]
                generation = output.get("generation", {})
                agent_logs = output.get("agent_logs", [])

                pred = extract_answer(generated_text, task_type=task)
                gold = batch["answers"][0].strip()
                is_correct = pred.strip() == gold.strip()

                local_update = {
                    "question": batch["questions"][0],
                    "gold": gold,
                    "prediction": pred,
                    "generated_text": generated_text[:500],
                    "generation": generation,
                    "correct": is_correct,
                    "agent_log": build_agent_sample_log(
                        question=batch["questions"][0],
                        gold=gold,
                        prediction=pred,
                        generated_text=generated_text,
                        generation=generation,
                        correct=is_correct,
                        agent_logs=agent_logs,
                    ) if write_agent_logs else None,
                    "sample_seconds": time.time() - t0,
                }

            gathered_updates = gather_sharded_objects(local_update, rank, world_size)

            if not is_main_process(rank):
                continue

            for shard_update in gathered_updates:
                if shard_update is None:
                    continue
                if shard_update["correct"]:
                    correct += 1
                total += 1
                sample_result = {
                    "question": shard_update["question"],
                    "gold": shard_update["gold"],
                    "prediction": shard_update["prediction"],
                    "generated_text": shard_update["generated_text"],
                    "generation": shard_update["generation"],
                    "correct": shard_update["correct"],
                }
                if write_agent_logs and shard_update["agent_log"] is not None:
                    sample_result["agent_log"] = shard_update["agent_log"]
                    agent_log_results.append(shard_update["agent_log"])
                results.append(sample_result)

            write_eval_snapshot(
                eval_path=eval_path,
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
                    "inference_mode": inference_mode,
                    "use_terminal_prefix": use_terminal_prefix,
                    "do_sample": do_sample,
                    "write_agent_logs": write_agent_logs,
                    "config": copy.deepcopy(config),
                },
                world_size=world_size,
                samples=results,
            )
            if write_agent_logs:
                write_agent_log_snapshot(
                    agent_log_path=agent_log_path,
                    method="ours_trained_multi_agent",
                    task=task,
                    parameters={
                        "config_path": config_path,
                        "checkpoint_path": checkpoint_path,
                        "max_samples": max_samples,
                        "generation_max_new_tokens": generation_max_new_tokens,
                        "inference_mode": inference_mode,
                        "use_terminal_prefix": use_terminal_prefix,
                        "do_sample": do_sample,
                        "write_agent_logs": write_agent_logs,
                    },
                    samples=agent_log_results,
                )

            if total > 0 and ((total % 10) == 0 or total == len(full_dataset)):
                acc = correct / total * 100
                print(f"  [{total}/{len(full_dataset)}] Acc: {acc:.1f}% ({correct}/{total})")

            if total <= 5 and results:
                latest = results[-1]
                status = "✓" if latest["correct"] else "✗"
                print(f"\n  Example {total} [{status}]:")
                print(f"    Q: {latest['question'][:100]}...")
                print(f"    Gold: {latest['gold']}")
                print(f"    Pred: {latest['prediction']}")
                print(f"    Gen:  {latest['generated_text'][:200]}")

    if not is_main_process(rank):
        if is_dist:
            dist.barrier()
            cleanup_eval_distributed()
        return

    # ── Summary ──
    all_results = results
    t_total = time.time() - t_start
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

    write_eval_snapshot(
        eval_path=eval_path,
        method="ours_trained_multi_agent",
        task=task,
        correct=correct,
        total=total,
        time_seconds=t_total,
        parameters={
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "max_samples": max_samples,
            "generation_max_new_tokens": generation_max_new_tokens,
            "inference_mode": inference_mode,
            "use_terminal_prefix": use_terminal_prefix,
            "do_sample": do_sample,
            "write_agent_logs": write_agent_logs,
            "config": copy.deepcopy(config),
        },
        world_size=world_size,
        samples=all_results,
    )
    print(f"  Results saved: {eval_path}")
    if write_agent_logs:
        print(f"  Agent logs saved: {agent_log_path}")

    if not run_baseline:
        if is_dist:
            dist.barrier()
        cleanup_eval_distributed()
        return

    # ── Also run baseline (no multi-agent, just the model) ──
    print(f"\n{'='*60}")
    print(f"  BASELINE (single model, no multi-agent)")
    print(f"{'='*60}")

    baseline_correct = 0
    baseline_total = 0
    baseline_samples = []

    dataloader_iter = iter(dataloader)
    for idx in range(max_steps):
        baseline_update = None
        if idx < local_steps:
            batch = next(dataloader_iter)
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=config["training"].get("max_seq_len", 2048),
            )
            task_ids = tokenized["input_ids"].to(device)
            task_mask = tokenized["attention_mask"].to(device)

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
            baseline_update = {
                "question": batch["questions"][0],
                "gold": gold,
                "prediction": pred,
                "generated_text": baseline_text[:500],
                "generation": generation,
                "correct": pred.strip() == gold.strip(),
            }

        gathered_baseline_updates = gather_sharded_objects(baseline_update, rank, world_size)
        if not is_main_process(rank):
            continue
        for shard_update in gathered_baseline_updates:
            if shard_update is None:
                continue
            if shard_update["correct"]:
                baseline_correct += 1
            baseline_total += 1
            baseline_samples.append(shard_update)

        baseline_acc = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0.0
        write_eval_snapshot(
            eval_path=eval_path,
            method="ours_trained_multi_agent",
            task=task,
            correct=correct,
            total=total,
            time_seconds=t_total,
            parameters={
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "max_samples": max_samples,
                "generation_max_new_tokens": generation_max_new_tokens,
                "inference_mode": inference_mode,
                "use_terminal_prefix": use_terminal_prefix,
                "do_sample": do_sample,
                "write_agent_logs": write_agent_logs,
                "config": copy.deepcopy(config),
            },
            world_size=world_size,
            samples=all_results,
            baseline_single_model={
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
            },
            comparison={
                "baseline_accuracy": baseline_acc,
                "improvement": accuracy - baseline_acc,
            },
        )

        if baseline_total > 0 and (baseline_total % 50) == 0:
            print(f"  [{baseline_total}/{len(full_dataset)}] Baseline Acc: {baseline_acc:.1f}%")

    baseline_acc = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Multi-agent (trained): {accuracy:.2f}%")
    print(f"  Single model baseline: {baseline_acc:.2f}%")
    print(f"  Improvement:           {accuracy - baseline_acc:+.2f}%")
    print(f"{'='*60}")

    # Append baseline to results
    write_eval_snapshot(
        eval_path=eval_path,
        method="ours_trained_multi_agent",
        task=task,
        correct=correct,
        total=total,
        time_seconds=t_total,
        parameters={
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "max_samples": max_samples,
            "generation_max_new_tokens": generation_max_new_tokens,
            "inference_mode": inference_mode,
            "use_terminal_prefix": use_terminal_prefix,
            "do_sample": do_sample,
            "write_agent_logs": write_agent_logs,
            "config": copy.deepcopy(config),
        },
        world_size=world_size,
        samples=all_results,
        baseline_single_model={
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
        },
        comparison={
            "baseline_accuracy": baseline_acc,
            "improvement": accuracy - baseline_acc,
        },
    )

    if is_dist:
        dist.barrier()
    cleanup_eval_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="chat_with_prefix",
        choices=["chat_with_prefix", "legacy_plain_with_prefix"],
    )
    parser.add_argument(
        "--no-terminal-prefix",
        action="store_true",
        help="Disable upstream latent prefix for the terminal agent during eval.",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Also run the embedded single-model baseline after ours eval.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-agent-logs", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.max_samples,
        args.max_new_tokens,
        inference_mode=args.inference_mode,
        use_terminal_prefix=not args.no_terminal_prefix,
        run_baseline=args.run_baseline,
        do_sample=args.do_sample,
        write_agent_logs=not args.no_agent_logs,
    )
