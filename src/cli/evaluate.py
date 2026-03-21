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
import hashlib
import json
import os
import shutil
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
    question_id: str,
    question: str,
    gold: str,
    prediction: str,
    generation: dict,
    correct: bool,
    agent_logs: list[dict],
) -> dict:
    return {
        "question_id": question_id,
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "generation": generation,
        "correct": correct,
        "agents": agent_logs,
    }


def _select_batch_item(value, index: int):
    if isinstance(value, list):
        return value[index]
    return value


def write_eval_snapshot(
    eval_path: Path,
    method: str,
    task: str,
    correct: int,
    total: int,
    time_seconds: float,
    *,
    avg_sample_seconds: float | None = None,
    avg_generated_tokens: float | None = None,
    avg_tokens_per_second: float | None = None,
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
            "avg_sample_seconds": avg_sample_seconds,
            "avg_generated_tokens": avg_generated_tokens,
            "avg_tokens_per_second": avg_tokens_per_second,
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


def write_role_agent_log_snapshots(agent_log_dir: Path, samples: list[dict]) -> None:
    role_payloads: dict[str, dict] = {}
    agent_log_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        question_id = str(sample["question_id"])
        for agent in sample.get("agents", []):
            role_name = agent["role_name"]
            payload = role_payloads.setdefault(
                role_name,
                {
                    "role_name": role_name,
                    "samples": {},
                },
            )
            payload["samples"][question_id] = {
                "question_id": question_id,
                "question": sample["question"],
                "input": {
                    "system_prompt": agent.get("system_prompt"),
                    "received_upstream_prefix": agent.get("received_upstream_prefix"),
                    "upstream_prefix": agent.get("upstream_prefix"),
                    "received_upstream_texts": agent.get("received_upstream_texts"),
                    "upstream_texts": agent.get("upstream_texts"),
                },
                "output": {
                    "output_type": agent.get("output_type"),
                    "generated_text": agent.get("generated_text"),
                    "generation": agent.get("generation"),
                    "hidden_trajectory": agent.get("hidden_trajectory"),
                    "compressed_prefix": agent.get("compressed_prefix"),
                },
            }

    for role_name, payload in role_payloads.items():
        role_path = agent_log_dir / f"{role_name}.json"
        with open(role_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def collate_fn(batch: list[dict]) -> dict:
    return {
        "question_ids": [item["question_id"] for item in batch],
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


def build_single_question_dataset(question: str) -> list[dict]:
    normalized_question = question.strip()
    question_hash = hashlib.sha1(normalized_question.encode("utf-8")).hexdigest()[:12]
    return [
        {
            "question_id": f"manual-{question_hash}",
            "question": normalized_question,
            "answer": "",
        }
    ]


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


def evaluate_loaded_system(
    system,
    config: dict,
    *,
    config_path: str,
    output_dir: str | Path,
    checkpoint_path: str | None = None,
    max_samples: int | None = None,
    split: str = "test",
    max_new_tokens: int = 2048,
    inference_mode: str = "chat_with_prefix",
    use_terminal_prefix: bool = True,
    run_baseline: bool = False,
    do_sample: bool = False,
    write_agent_logs: bool = True,
    worker: int | None = None,
    batch_size: int = 1,
    device: torch.device | None = None,
    rank: int = 0,
    world_size: int = 1,
    is_dist: bool = False,
):
    generation_max_new_tokens = max_new_tokens
    output_dir = Path(output_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(system, "to"):
        system.to(device)
    if hasattr(system, "eval"):
        system.eval()

    if is_main_process(rank):
        print(f"\nLearned adjacency:")
        if hasattr(system, "log_adjacency"):
            print(system.log_adjacency())
        if hasattr(system, "adjacency"):
            A = system.adjacency.get_adjacency().detach()
            print(f"Range: [{A.min().item():.4f}, {A.max().item():.4f}]")
            print(f"Hard adjacency (threshold=0.5):")
            print(system.adjacency.get_hard_adjacency())

    task = config["training"]["task"]
    if max_samples is not None and max_samples < 0:
        max_samples = None
    if is_main_process(rank):
        print(f"\nLoading {task} {split} set...")
    full_dataset = create_dataset(task=task, split=split, max_samples=max_samples)
    dataset = shard_dataset(full_dataset, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    if is_main_process(rank):
        print(f"{split.capitalize()} samples: {len(full_dataset)}")
        if world_size > 1:
            print(
                f"Sharded samples per rank: "
                f"{[len(range(r, len(full_dataset), world_size)) for r in range(world_size)]}"
            )
        print(f"Eval batch size per rank: {batch_size}")

    correct = 0
    total = 0
    results = []
    agent_log_results = []
    sample_durations = []
    generated_token_counts = []
    eval_stem = "eval_results" if split == "test" else f"eval_results_{split}"
    agent_log_stem = "agent_logs" if split == "test" else f"agent_logs_{split}"
    eval_path = output_dir / f"{eval_stem}.json"
    agent_log_path = output_dir / f"{agent_log_stem}.json"
    role_agent_log_dir = output_dir / "agent_log"
    if is_main_process(rank):
        for path in (eval_path, agent_log_path):
            if path.exists():
                path.unlink()
        if role_agent_log_dir.exists():
            shutil.rmtree(role_agent_log_dir)

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
                batch_questions = batch["questions"]
                batch_question_ids = batch["question_ids"]
                batch_answers = batch["answers"]
                batch_generated_text = generated_text if isinstance(generated_text, list) else [generated_text]
                batch_size_actual = len(batch_questions)
                batch_elapsed = time.time() - t0
                per_sample_seconds = batch_elapsed / max(batch_size_actual, 1)
                local_update = []
                for sample_idx in range(batch_size_actual):
                    sample_generation = {
                        key: _select_batch_item(value, sample_idx)
                        for key, value in generation.items()
                    }
                    pred = extract_answer(batch_generated_text[sample_idx], task_type=task)
                    gold = batch_answers[sample_idx].strip()
                    is_correct = pred.strip() == gold.strip()
                    local_update.append(
                        {
                            "question": batch_questions[sample_idx],
                            "question_id": batch_question_ids[sample_idx],
                            "gold": gold,
                            "prediction": pred,
                            "generation": sample_generation,
                            "correct": is_correct,
                            "agent_log": build_agent_sample_log(
                                question_id=batch_question_ids[sample_idx],
                                question=batch_questions[sample_idx],
                                gold=gold,
                                prediction=pred,
                                generation=sample_generation,
                                correct=is_correct,
                                agent_logs=agent_logs,
                            ) if write_agent_logs else None,
                            "sample_seconds": per_sample_seconds,
                        }
                    )

            gathered_updates = gather_sharded_objects(local_update, rank, world_size)

            if not is_main_process(rank):
                continue

            for shard_updates in gathered_updates:
                if shard_updates is None:
                    continue
                for shard_update in shard_updates:
                    if shard_update["correct"]:
                        correct += 1
                    total += 1
                    sample_durations.append(shard_update["sample_seconds"])
                    generated_token_counts.append(shard_update["generation"].get("generated_token_count", 0))
                    sample_result = {
                        "question": shard_update["question"],
                        "question_id": shard_update["question_id"],
                        "gold": shard_update["gold"],
                        "prediction": shard_update["prediction"],
                        "generation": shard_update["generation"],
                        "correct": shard_update["correct"],
                    }
                    if write_agent_logs and shard_update["agent_log"] is not None:
                        sample_result["agent_log"] = shard_update["agent_log"]
                        agent_log_results.append(shard_update["agent_log"])
                    results.append(sample_result)

            params = {
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "split": split,
                "max_samples": max_samples,
                "generation_max_new_tokens": generation_max_new_tokens,
                "inference_mode": inference_mode,
                "use_terminal_prefix": use_terminal_prefix,
                "do_sample": do_sample,
                "write_agent_logs": write_agent_logs,
                "worker": worker,
                "batch_size": batch_size,
                "config": copy.deepcopy(config),
            }
            write_eval_snapshot(
                eval_path=eval_path,
                method="ours_trained_multi_agent",
                task=task,
                correct=correct,
                total=total,
                time_seconds=time.time() - t_start,
                avg_sample_seconds=(sum(sample_durations) / len(sample_durations)) if sample_durations else None,
                avg_generated_tokens=(
                    sum(generated_token_counts) / len(generated_token_counts)
                ) if generated_token_counts else None,
                avg_tokens_per_second=(
                    (sum(generated_token_counts) / sum(sample_durations))
                    if sample_durations and sum(sample_durations) > 0
                    else None
                ),
                parameters=params,
                world_size=world_size,
                samples=results,
            )
            if write_agent_logs:
                write_agent_log_snapshot(
                    agent_log_path=agent_log_path,
                    method="ours_trained_multi_agent",
                    task=task,
                    parameters=params,
                    samples=agent_log_results,
                )
                write_role_agent_log_snapshots(role_agent_log_dir, agent_log_results)

            if total > 0 and ((total % 10) == 0 or total == len(full_dataset)):
                acc = correct / total * 100
                print(f"  [{total}/{len(full_dataset)}] Acc: {acc:.1f}% ({correct}/{total})")

    if not is_main_process(rank):
        return None

    t_total = time.time() - t_start
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Task:     {task}")
    print(f"  Split:    {split}")
    print(f"  Samples:  {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Time:     {t_total:.1f}s ({t_total/max(total, 1):.1f}s/sample)")
    if sample_durations:
        print(f"  Avg sample: {sum(sample_durations)/len(sample_durations):.2f}s")
    if generated_token_counts:
        print(f"  Avg generated tokens: {sum(generated_token_counts)/len(generated_token_counts):.2f}")
        if sample_durations and sum(sample_durations) > 0:
            print(f"  Tokens/sec: {sum(generated_token_counts)/sum(sample_durations):.2f}")
    print(f"{'='*60}")

    if run_baseline:
        print("Baseline comparison is not supported in evaluate_loaded_system().")

    result_payload = {
        "method": "ours_trained_multi_agent",
        "task": task,
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "time_seconds": t_total,
            "avg_sample_seconds": (sum(sample_durations) / len(sample_durations)) if sample_durations else None,
            "avg_generated_tokens": (
                sum(generated_token_counts) / len(generated_token_counts)
            ) if generated_token_counts else None,
            "avg_tokens_per_second": (
                (sum(generated_token_counts) / sum(sample_durations))
                if sample_durations and sum(sample_durations) > 0
                else None
            ),
        },
        "paths": {
            "eval_path": str(eval_path),
            "agent_log_path": str(agent_log_path) if write_agent_logs else None,
            "role_agent_log_dir": str(role_agent_log_dir) if write_agent_logs else None,
        },
    }
    return result_payload


def evaluate(
    config_path: str,
    checkpoint_path: str,
    max_samples: int | None = None,
    split: str = "test",
    max_new_tokens: int = 2048,
    inference_mode: str = "chat_with_prefix",
    use_terminal_prefix: bool = True,
    run_baseline: bool = False,
    do_sample: bool = False,
    write_agent_logs: bool = True,
    worker: int | None = None,
    batch_size: int = 1,
    question: str | None = None,
    output_dir: str | Path | None = None,
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

    base_model_state = ckpt.get("base_model_state")
    if base_model_state is not None:
        system.base_model.model.load_state_dict(base_model_state)

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
    if question is not None:
        if is_main_process(rank):
            print("\nLoading manual inference question...")
        full_dataset = build_single_question_dataset(question)
    else:
        if is_main_process(rank):
            print(f"\nLoading {task} test set...")
        full_dataset = create_dataset(task=task, split=split, max_samples=max_samples)
    dataset = shard_dataset(full_dataset, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    if is_main_process(rank):
        print(f"{split.capitalize()} samples: {len(full_dataset)}")
        if world_size > 1:
            print(f"Sharded samples per rank: {[len(range(r, len(full_dataset), world_size)) for r in range(world_size)]}")
        print(f"Eval batch size per rank: {batch_size}")

    # ── Evaluate ──
    correct = 0
    total = 0
    results = []
    agent_log_results = []
    sample_durations = []
    generated_token_counts = []
    checkpoint_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    eval_stem = "eval_results" if split == "test" else f"eval_results_{split}"
    agent_log_stem = "agent_logs" if split == "test" else f"agent_logs_{split}"
    eval_path = checkpoint_dir / f"{eval_stem}.json"
    agent_log_path = checkpoint_dir / f"{agent_log_stem}.json"
    role_agent_log_dir = checkpoint_dir / "agent_log"
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
        if role_agent_log_dir.exists():
            shutil.rmtree(role_agent_log_dir)

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
                batch_questions = batch["questions"]
                batch_question_ids = batch["question_ids"]
                batch_answers = batch["answers"]
                batch_generated_text = generated_text if isinstance(generated_text, list) else [generated_text]
                batch_size_actual = len(batch_questions)
                batch_elapsed = time.time() - t0
                per_sample_seconds = batch_elapsed / max(batch_size_actual, 1)
                local_update = []
                for sample_idx in range(batch_size_actual):
                    sample_generation = {
                        key: _select_batch_item(value, sample_idx)
                        for key, value in generation.items()
                    }
                    pred = extract_answer(batch_generated_text[sample_idx], task_type=task)
                    gold = batch_answers[sample_idx].strip()
                    is_correct = pred.strip() == gold.strip()
                    local_update.append(
                        {
                            "question": batch_questions[sample_idx],
                            "question_id": batch_question_ids[sample_idx],
                            "gold": gold,
                            "prediction": pred,
                            "generation": sample_generation,
                            "correct": is_correct,
                            "agent_log": build_agent_sample_log(
                                question_id=batch_question_ids[sample_idx],
                                question=batch_questions[sample_idx],
                                gold=gold,
                                prediction=pred,
                                generation=sample_generation,
                                correct=is_correct,
                                agent_logs=agent_logs,
                            ) if write_agent_logs else None,
                            "sample_seconds": per_sample_seconds,
                        }
                    )

            gathered_updates = gather_sharded_objects(local_update, rank, world_size)

            if not is_main_process(rank):
                continue

            for shard_updates in gathered_updates:
                if shard_updates is None:
                    continue
                for shard_update in shard_updates:
                    if shard_update["correct"]:
                        correct += 1
                    total += 1
                    sample_durations.append(shard_update["sample_seconds"])
                    generated_token_counts.append(shard_update["generation"].get("generated_token_count", 0))
                    sample_result = {
                        "question": shard_update["question"],
                        "question_id": shard_update["question_id"],
                        "gold": shard_update["gold"],
                        "prediction": shard_update["prediction"],
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
                avg_sample_seconds=(sum(sample_durations) / len(sample_durations)) if sample_durations else None,
                avg_generated_tokens=(sum(generated_token_counts) / len(generated_token_counts)) if generated_token_counts else None,
                avg_tokens_per_second=(
                    (sum(generated_token_counts) / sum(sample_durations))
                    if sample_durations and sum(sample_durations) > 0
                    else None
                ),
                parameters={
                    "config_path": config_path,
                    "checkpoint_path": checkpoint_path,
                    "split": split,
                    "max_samples": max_samples,
                    "question": question,
                    "generation_max_new_tokens": generation_max_new_tokens,
                    "inference_mode": inference_mode,
                    "use_terminal_prefix": use_terminal_prefix,
                    "do_sample": do_sample,
                    "write_agent_logs": write_agent_logs,
                    "worker": worker,
                    "batch_size": batch_size,
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
                        "split": split,
                        "max_samples": max_samples,
                        "generation_max_new_tokens": generation_max_new_tokens,
                        "inference_mode": inference_mode,
                        "use_terminal_prefix": use_terminal_prefix,
                        "do_sample": do_sample,
                        "write_agent_logs": write_agent_logs,
                        "worker": worker,
                        "batch_size": batch_size,
                    },
                    samples=agent_log_results,
                )
                write_role_agent_log_snapshots(role_agent_log_dir, agent_log_results)

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
                print(f"    Gen:  {latest['generation'].get('generated_text', '')[:200]}")

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
    print(f"  Split:    {split}")
    print(f"  Samples:  {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Time:     {t_total:.1f}s ({t_total/total:.1f}s/sample)")
    if sample_durations:
        print(f"  Avg sample: {sum(sample_durations)/len(sample_durations):.2f}s")
    if generated_token_counts:
        avg_tokens = sum(generated_token_counts) / len(generated_token_counts)
        print(f"  Avg gen tokens: {avg_tokens:.1f}")
        if sum(sample_durations) > 0:
            print(f"  Tokens/s: {sum(generated_token_counts)/sum(sample_durations):.2f}")
    print(f"{'='*60}")

    write_eval_snapshot(
        eval_path=eval_path,
        method="ours_trained_multi_agent",
        task=task,
        correct=correct,
        total=total,
        time_seconds=t_total,
        avg_sample_seconds=(sum(sample_durations) / len(sample_durations)) if sample_durations else None,
        avg_generated_tokens=(sum(generated_token_counts) / len(generated_token_counts)) if generated_token_counts else None,
        avg_tokens_per_second=(
            (sum(generated_token_counts) / sum(sample_durations))
            if sample_durations and sum(sample_durations) > 0
            else None
        ),
        parameters={
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "split": split,
            "max_samples": max_samples,
            "question": question,
            "generation_max_new_tokens": generation_max_new_tokens,
            "inference_mode": inference_mode,
            "use_terminal_prefix": use_terminal_prefix,
            "do_sample": do_sample,
            "write_agent_logs": write_agent_logs,
            "worker": worker,
            "batch_size": batch_size,
            "config": copy.deepcopy(config),
        },
        world_size=world_size,
        samples=all_results,
    )
    print(f"  Results saved: {eval_path}")
    if write_agent_logs:
        print(f"  Agent logs saved: {agent_log_path}")
        print(f"  Role agent logs saved: {role_agent_log_dir}")

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
                "question_id": batch["question_ids"][0],
                "question": batch["questions"][0],
                "gold": gold,
                "prediction": pred,
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
            avg_sample_seconds=(sum(sample_durations) / len(sample_durations)) if sample_durations else None,
            avg_generated_tokens=(sum(generated_token_counts) / len(generated_token_counts)) if generated_token_counts else None,
            avg_tokens_per_second=(
                (sum(generated_token_counts) / sum(sample_durations))
                if sample_durations and sum(sample_durations) > 0
                else None
            ),
            parameters={
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "split": split,
                "max_samples": max_samples,
                "question": question,
                "generation_max_new_tokens": generation_max_new_tokens,
                "inference_mode": inference_mode,
                "use_terminal_prefix": use_terminal_prefix,
                "do_sample": do_sample,
                "write_agent_logs": write_agent_logs,
                "worker": worker,
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
        avg_sample_seconds=(sum(sample_durations) / len(sample_durations)) if sample_durations else None,
        avg_generated_tokens=(sum(generated_token_counts) / len(generated_token_counts)) if generated_token_counts else None,
        avg_tokens_per_second=(
            (sum(generated_token_counts) / sum(sample_durations))
            if sample_durations and sum(sample_durations) > 0
            else None
        ),
        parameters={
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "split": split,
            "max_samples": max_samples,
            "question": question,
            "generation_max_new_tokens": generation_max_new_tokens,
            "inference_mode": inference_mode,
            "use_terminal_prefix": use_terminal_prefix,
            "do_sample": do_sample,
            "write_agent_logs": write_agent_logs,
            "worker": worker,
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
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="chat_with_prefix",
        choices=["chat_with_prefix", "legacy_plain_with_prefix", "chat_with_text"],
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
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.max_samples,
        args.split,
        args.max_new_tokens,
        inference_mode=args.inference_mode,
        use_terminal_prefix=not args.no_terminal_prefix,
        run_baseline=args.run_baseline,
        do_sample=args.do_sample,
        write_agent_logs=not args.no_agent_logs,
        worker=args.worker,
        batch_size=args.batch_size,
        question=args.question,
        output_dir=args.output_dir,
    )
