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
import inspect
import json
import os
import re
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
from src.utils.progress_logging import ProgressLogger
from src.utils.answer_extraction import extract_answer
from src.pipeline.multi_agent_system import MultiAgentSystem
from data.dataset import create_dataset
from src.models.agent import Agent


def build_agent_sample_log(
    question: str,
    gold: str,
    prediction: str,
    generation: dict,
    correct: bool,
    agent_logs: list[dict],
) -> dict:
    return {
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


def _select_agent_log_value(value, index: int):
    if isinstance(value, list):
        return value[index]
    if isinstance(value, dict):
        return {
            key: _select_agent_log_value(nested_value, index)
            for key, nested_value in value.items()
        }
    return value


def select_agent_logs_for_sample(agent_logs: list[dict], index: int) -> list[dict]:
    sample_logs = []
    for agent_log in agent_logs:
        sample_log = copy.deepcopy(agent_log)
        if sample_log.get("output_type") in {"text", "text_message"}:
            generated_text = sample_log.get("generated_text")
            if isinstance(generated_text, list) and index < len(generated_text):
                sample_log["generated_text"] = generated_text[index]
            generation = sample_log.get("generation")
            if isinstance(generation, dict):
                sample_log["generation"] = {
                    key: _select_agent_log_value(value, index)
                    for key, value in generation.items()
                }
            upstream_text_messages = sample_log.get("upstream_text_messages")
            if isinstance(upstream_text_messages, list):
                if index < len(upstream_text_messages) and isinstance(upstream_text_messages[index], list):
                    sample_log["upstream_text_messages"] = upstream_text_messages[index]
        sample_logs.append(sample_log)
    return sample_logs


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


def build_text_preview(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def is_suspected_gibberish(text: str) -> bool:
    if not text:
        return False
    if "\ufffd" in text or "ДД" in text or "TableTableTable" in text:
        return True

    compact = " ".join(text.split())
    if len(compact) < 80:
        return False

    repeated_words = re.findall(r"([A-Za-z]{3,})\1{3,}", compact)
    if repeated_words:
        return True

    tokens = re.findall(r"[A-Za-z\u0400-\u04ff\u4e00-\u9fff_]+", compact)
    if tokens:
        most_common = max(tokens.count(token) for token in set(tokens))
        if most_common >= 12 and (most_common / len(tokens)) >= 0.25:
            return True

    punctuation_ratio = sum(char in ";\n\t" for char in text) / max(len(text), 1)
    return punctuation_ratio > 0.18 and "####" not in text and "answer" not in text.lower()


def print_sample_preview(
    sample_number: int,
    question: str,
    gold: str,
    prediction: str,
    generation: dict,
    logger: ProgressLogger | None = None,
) -> bool:
    generated_text = generation.get("generated_text", "") or ""
    suspected_gibberish = is_suspected_gibberish(generated_text)
    status = "WARN" if suspected_gibberish else "OK"
    log = logger.log if logger is not None else print
    log(f"\n  Sample {sample_number} [{status}]")
    log(f"    Q: {build_text_preview(question, max_chars=140)}")
    log(f"    Gold: {gold}")
    log(f"    Pred: {prediction or '<empty>'}")
    log(f"    Finish: {generation.get('finish_reason')} | Tokens: {generation.get('generated_token_count')}")
    log(f"    Gen: {build_text_preview(generated_text, max_chars=240)}")
    return suspected_gibberish


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


def run_system_batch(
    system,
    config: dict,
    batch: dict,
    *,
    task: str,
    device: torch.device,
    generation_max_new_tokens: int,
    inference_mode: str,
    use_terminal_prefix: bool,
    communication_mode: str,
    text_message_edge_threshold: float,
    text_message_max_new_tokens: int,
    do_sample: bool,
    write_agent_logs: bool,
) -> tuple[list[dict], float]:
    t0 = time.time()
    tokenized = system.base_model.tokenize(
        batch["questions"],
        max_length=config["training"].get("max_seq_len", 2048),
    )
    task_ids = tokenized["input_ids"].to(device)
    task_mask = tokenized["attention_mask"].to(device)

    call_kwargs = {
        "task_token_ids": task_ids,
        "task_attention_mask": task_mask,
        "max_new_tokens": generation_max_new_tokens,
        "inference_mode": inference_mode,
        "use_terminal_prefix": use_terminal_prefix,
        "communication_mode": communication_mode,
        "text_message_edge_threshold": text_message_edge_threshold,
        "text_message_max_new_tokens": text_message_max_new_tokens,
        "do_sample": do_sample,
        "collect_agent_logs": write_agent_logs,
    }
    call_target = system.forward if hasattr(system, "forward") else system.__call__
    supported_params = set(inspect.signature(call_target).parameters)
    filtered_kwargs = {
        key: value for key, value in call_kwargs.items()
        if key in supported_params
    }
    output = system(**filtered_kwargs)

    generated_text = output["generated_text"]
    generation = output.get("generation", {})
    agent_logs = output.get("agent_logs", [])
    batch_questions = batch["questions"]
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
                "gold": gold,
                "prediction": pred,
                "generation": sample_generation,
                "correct": is_correct,
                "agent_log": build_agent_sample_log(
                    question=batch_questions[sample_idx],
                    gold=gold,
                    prediction=pred,
                    generation=sample_generation,
                    correct=is_correct,
                    agent_logs=select_agent_logs_for_sample(agent_logs, sample_idx),
                ) if write_agent_logs else None,
                "sample_seconds": per_sample_seconds,
            }
        )
    return local_update, batch_elapsed


def run_preflight_preview(
    system,
    config: dict,
    dataset,
    *,
    task: str,
    device: torch.device,
    rank: int,
    world_size: int,
    generation_max_new_tokens: int,
    inference_mode: str,
    use_terminal_prefix: bool,
    communication_mode: str,
    text_message_edge_threshold: float,
    text_message_max_new_tokens: int,
    do_sample: bool,
    write_agent_logs: bool,
    preview_limit: int = 5,
    logger: ProgressLogger | None = None,
) -> tuple[int, int]:
    if preview_limit <= 0:
        return 0, 0
    local_preview_count = min(preview_limit, len(dataset))
    preview_counts = gather_sharded_objects(local_preview_count, rank, world_size)
    max_preview_steps = max(preview_counts) if preview_counts else 0
    preview_printed = 0
    suspected_gibberish_count = 0

    log = logger.log if logger is not None else print
    if is_main_process(rank) and max_preview_steps > 0:
        log(f"\nRunning preflight preview on first {min(sum(preview_counts), preview_limit * world_size)} local samples...")

    with torch.no_grad():
        for idx in range(max_preview_steps):
            local_update = None
            if idx < local_preview_count:
                sample = dataset[idx]
                batch = {
                    "questions": [sample["question"]],
                    "answers": [sample["answer"]],
                }
                local_update, _ = run_system_batch(
                    system,
                    config,
                    batch,
                    task=task,
                    device=device,
                    generation_max_new_tokens=generation_max_new_tokens,
                    inference_mode=inference_mode,
                    use_terminal_prefix=use_terminal_prefix,
                    communication_mode=communication_mode,
                    text_message_edge_threshold=text_message_edge_threshold,
                    text_message_max_new_tokens=text_message_max_new_tokens,
                    do_sample=do_sample,
                    write_agent_logs=write_agent_logs,
                )
            gathered_updates = gather_sharded_objects(local_update, rank, world_size)
            if not is_main_process(rank):
                continue
            for shard_updates in gathered_updates:
                if not shard_updates:
                    continue
                for shard_update in shard_updates:
                    if preview_printed >= preview_limit:
                        break
                    if print_sample_preview(
                        sample_number=preview_printed + 1,
                        question=shard_update["question"],
                        gold=shard_update["gold"],
                        prediction=shard_update["prediction"],
                        generation=shard_update["generation"],
                        logger=logger,
                    ):
                        suspected_gibberish_count += 1
                    preview_printed += 1
    return preview_printed, suspected_gibberish_count


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
    communication_mode: str = "latent_prefix",
    text_message_edge_threshold: float = 0.5,
    text_message_max_new_tokens: int = 512,
    run_baseline: bool = False,
    do_sample: bool = False,
    write_agent_logs: bool = True,
    worker: int | None = None,
    batch_size: int = 1,
    preview_limit: int = 5,
    device: torch.device | None = None,
    rank: int = 0,
    world_size: int = 1,
    is_dist: bool = False,
):
    generation_max_new_tokens = max_new_tokens
    output_dir = Path(output_dir)
    progress_logger = ProgressLogger(output_dir / "eval_progress.log")
    log = progress_logger.log
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(system, "to"):
        system.to(device)
    if hasattr(system, "eval"):
        system.eval()

    if is_main_process(rank):
        log(f"\nLearned adjacency:")
        if hasattr(system, "log_adjacency"):
            log(str(system.log_adjacency()))
        if hasattr(system, "adjacency"):
            A = system.adjacency.get_adjacency().detach()
            log(f"Range: [{A.min().item():.4f}, {A.max().item():.4f}]")
            log("Hard adjacency (threshold=0.5):")
            log(str(system.adjacency.get_hard_adjacency()))

    task = config["training"]["task"]
    if max_samples is not None and max_samples < 0:
        max_samples = None
    if is_main_process(rank):
        log(f"\nLoading {task} {split} set...")
    full_dataset = create_dataset(task=task, split=split, max_samples=max_samples)
    dataset = shard_dataset(full_dataset, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    if is_main_process(rank):
        log(f"{split.capitalize()} samples: {len(full_dataset)}")
        if world_size > 1:
            log(
                f"Sharded samples per rank: "
                f"{[len(range(r, len(full_dataset), world_size)) for r in range(world_size)]}"
            )
        log(f"Eval batch size per rank: {batch_size}")

    correct = 0
    total = 0
    results = []
    agent_log_results = []
    sample_durations = []
    generated_token_counts = []
    preview_printed = 0
    suspected_gibberish_count = 0
    eval_stem = "eval_results" if split == "test" else f"eval_results_{split}"
    agent_log_stem = "agent_logs" if split == "test" else f"agent_logs_{split}"
    eval_path = output_dir / f"{eval_stem}.json"
    agent_log_path = output_dir / f"{agent_log_stem}.json"
    if is_main_process(rank):
        for path in (eval_path, agent_log_path):
            if path.exists():
                path.unlink()

    if is_main_process(rank):
        log("\nRunning evaluation...")
    t_start = time.time()

    preview_printed, suspected_gibberish_count = run_preflight_preview(
        system,
        config,
        dataset,
        task=task,
        device=device,
        rank=rank,
        world_size=world_size,
        generation_max_new_tokens=generation_max_new_tokens,
        inference_mode=inference_mode,
        use_terminal_prefix=use_terminal_prefix,
        communication_mode=communication_mode,
        text_message_edge_threshold=text_message_edge_threshold,
        text_message_max_new_tokens=text_message_max_new_tokens,
        do_sample=do_sample,
        write_agent_logs=write_agent_logs,
        preview_limit=preview_limit,
        logger=progress_logger,
    )

    local_steps = len(dataloader)
    step_counts = gather_sharded_objects(local_steps, rank, world_size)
    max_steps = max(step_counts)
    dataloader_iter = iter(dataloader)

    with torch.no_grad():
        for idx in range(max_steps):
            batch = None
            local_update = None
            if idx < local_steps:
                batch = next(dataloader_iter)
                local_update, _ = run_system_batch(
                    system,
                    config,
                    batch,
                    task=task,
                    device=device,
                    generation_max_new_tokens=generation_max_new_tokens,
                    inference_mode=inference_mode,
                    use_terminal_prefix=use_terminal_prefix,
                    communication_mode=communication_mode,
                    text_message_edge_threshold=text_message_edge_threshold,
                    text_message_max_new_tokens=text_message_max_new_tokens,
                    do_sample=do_sample,
                    write_agent_logs=write_agent_logs,
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
                "communication_mode": communication_mode,
                "text_message_edge_threshold": text_message_edge_threshold,
                "text_message_max_new_tokens": text_message_max_new_tokens,
                "do_sample": do_sample,
                "write_agent_logs": write_agent_logs,
                "worker": worker,
                "batch_size": batch_size,
                "preview_limit": preview_limit,
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

            if total > 0 and ((total % 10) == 0 or total == len(full_dataset)):
                acc = correct / total * 100
                log(f"  [{total}/{len(full_dataset)}] Acc: {acc:.1f}% ({correct}/{total})")

    if not is_main_process(rank):
        return None

    t_total = time.time() - t_start
    accuracy = correct / total * 100 if total > 0 else 0.0

    log(f"\n{'='*60}")
    log("  EVALUATION RESULTS")
    log(f"{'='*60}")
    log(f"  Task:     {task}")
    log(f"  Split:    {split}")
    log(f"  Samples:  {total}")
    log(f"  Correct:  {correct}")
    log(f"  Accuracy: {accuracy:.2f}%")
    log(f"  Time:     {t_total:.1f}s ({t_total/max(total, 1):.1f}s/sample)")
    if sample_durations:
        log(f"  Avg sample: {sum(sample_durations)/len(sample_durations):.2f}s")
    if generated_token_counts:
        log(f"  Avg generated tokens: {sum(generated_token_counts)/len(generated_token_counts):.2f}")
        if sample_durations and sum(sample_durations) > 0:
            log(f"  Tokens/sec: {sum(generated_token_counts)/sum(sample_durations):.2f}")
    if suspected_gibberish_count > 0:
        log(f"  Warning: suspected gibberish in {suspected_gibberish_count}/{preview_printed} previewed samples")
    log(f"{'='*60}")

    if run_baseline:
        log("Baseline comparison is not supported in evaluate_loaded_system().")

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
            "suspected_gibberish_previews": suspected_gibberish_count,
        },
        "paths": {
            "eval_path": str(eval_path),
            "agent_log_path": str(agent_log_path) if write_agent_logs else None,
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
    communication_mode: str = "latent_prefix",
    text_message_edge_threshold: float = 0.5,
    text_message_max_new_tokens: int = 512,
    run_baseline: bool = False,
    do_sample: bool = False,
    write_agent_logs: bool = True,
    worker: int | None = None,
    batch_size: int = 1,
    preview_limit: int = 5,
):
    config = load_config(config_path)
    generation_max_new_tokens = max_new_tokens
    device, rank, world_size, is_dist = setup_eval_distributed()
    checkpoint_dir = Path(checkpoint_path).parent
    progress_logger = ProgressLogger(checkpoint_dir / "eval_progress.log")
    log = progress_logger.log
    if is_main_process(rank):
        log(f"Device: {device}")
        log(f"World size: {world_size}")

    # ── Build system ──
    if is_main_process(rank):
        log("Building multi-agent system...")
    system = MultiAgentSystem(config)

    # ── Load checkpoint ──
    if is_main_process(rank):
        log(f"Loading checkpoint: {checkpoint_path}")
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
        log(f"\nLearned adjacency:")
        log(str(system.log_adjacency()))
        A = system.adjacency.get_adjacency().detach()
        log(f"Range: [{A.min().item():.4f}, {A.max().item():.4f}]")
        log("Hard adjacency (threshold=0.5):")
        log(str(system.adjacency.get_hard_adjacency()))

    # ── Dataset ──
    task = config["training"]["task"]
    if max_samples is not None and max_samples < 0:
        max_samples = None
    if is_main_process(rank):
        log(f"\nLoading {task} test set...")
    full_dataset = create_dataset(task=task, split=split, max_samples=max_samples)
    dataset = shard_dataset(full_dataset, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    if is_main_process(rank):
        log(f"{split.capitalize()} samples: {len(full_dataset)}")
        if world_size > 1:
            log(f"Sharded samples per rank: {[len(range(r, len(full_dataset), world_size)) for r in range(world_size)]}")
        log(f"Eval batch size per rank: {batch_size}")

    # ── Evaluate ──
    correct = 0
    total = 0
    results = []
    agent_log_results = []
    sample_durations = []
    generated_token_counts = []
    eval_stem = "eval_results" if split == "test" else f"eval_results_{split}"
    agent_log_stem = "agent_logs" if split == "test" else f"agent_logs_{split}"
    eval_path = checkpoint_dir / f"{eval_stem}.json"
    agent_log_path = checkpoint_dir / f"{agent_log_stem}.json"
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
        log("\nRunning evaluation...")
    t_start = time.time()

    preview_printed, suspected_gibberish_count = run_preflight_preview(
        system,
        config,
        dataset,
        task=task,
        device=device,
        rank=rank,
        world_size=world_size,
        generation_max_new_tokens=generation_max_new_tokens,
        inference_mode=inference_mode,
        use_terminal_prefix=use_terminal_prefix,
        communication_mode=communication_mode,
        text_message_edge_threshold=text_message_edge_threshold,
        text_message_max_new_tokens=text_message_max_new_tokens,
        do_sample=do_sample,
        write_agent_logs=write_agent_logs,
        preview_limit=preview_limit,
        logger=progress_logger,
    )

    local_steps = len(dataloader)
    step_counts = gather_sharded_objects(local_steps, rank, world_size)
    max_steps = max(step_counts)
    dataloader_iter = iter(dataloader)

    with torch.no_grad():
        for idx in range(max_steps):
            batch = None
            local_update = None
            if idx < local_steps:
                batch = next(dataloader_iter)
                local_update, _ = run_system_batch(
                    system,
                    config,
                    batch,
                    task=task,
                    device=device,
                    generation_max_new_tokens=generation_max_new_tokens,
                    inference_mode=inference_mode,
                    use_terminal_prefix=use_terminal_prefix,
                    communication_mode=communication_mode,
                    text_message_edge_threshold=text_message_edge_threshold,
                    text_message_max_new_tokens=text_message_max_new_tokens,
                    do_sample=do_sample,
                    write_agent_logs=write_agent_logs,
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
                    "generation_max_new_tokens": generation_max_new_tokens,
                    "inference_mode": inference_mode,
                    "use_terminal_prefix": use_terminal_prefix,
                    "communication_mode": communication_mode,
                    "text_message_edge_threshold": text_message_edge_threshold,
                    "text_message_max_new_tokens": text_message_max_new_tokens,
                    "do_sample": do_sample,
                    "write_agent_logs": write_agent_logs,
                    "worker": worker,
                    "batch_size": batch_size,
                    "preview_limit": preview_limit,
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
                        "communication_mode": communication_mode,
                        "text_message_edge_threshold": text_message_edge_threshold,
                        "text_message_max_new_tokens": text_message_max_new_tokens,
                        "do_sample": do_sample,
                        "write_agent_logs": write_agent_logs,
                        "worker": worker,
                        "batch_size": batch_size,
                        "preview_limit": preview_limit,
                    },
                    samples=agent_log_results,
                )

            if total > 0 and ((total % 10) == 0 or total == len(full_dataset)):
                acc = correct / total * 100
                log(f"  [{total}/{len(full_dataset)}] Acc: {acc:.1f}% ({correct}/{total})")

    if not is_main_process(rank):
        if is_dist:
            dist.barrier()
            cleanup_eval_distributed()
        return

    # ── Summary ──
    all_results = results
    t_total = time.time() - t_start
    accuracy = correct / total * 100 if total > 0 else 0.0

    log(f"\n{'='*60}")
    log("  EVALUATION RESULTS")
    log(f"{'='*60}")
    log(f"  Task:     {task}")
    log(f"  Split:    {split}")
    log(f"  Samples:  {total}")
    log(f"  Correct:  {correct}")
    log(f"  Accuracy: {accuracy:.2f}%")
    log(f"  Time:     {t_total:.1f}s ({t_total/total:.1f}s/sample)")
    if sample_durations:
        log(f"  Avg sample: {sum(sample_durations)/len(sample_durations):.2f}s")
    if generated_token_counts:
        avg_tokens = sum(generated_token_counts) / len(generated_token_counts)
        log(f"  Avg gen tokens: {avg_tokens:.1f}")
        if sum(sample_durations) > 0:
            log(f"  Tokens/s: {sum(generated_token_counts)/sum(sample_durations):.2f}")
    if suspected_gibberish_count > 0:
        log(f"  Warning: suspected gibberish in {suspected_gibberish_count}/{preview_printed} previewed samples")
    log(f"{'='*60}")

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
            "generation_max_new_tokens": generation_max_new_tokens,
            "inference_mode": inference_mode,
            "use_terminal_prefix": use_terminal_prefix,
            "communication_mode": communication_mode,
            "text_message_edge_threshold": text_message_edge_threshold,
            "text_message_max_new_tokens": text_message_max_new_tokens,
            "do_sample": do_sample,
            "write_agent_logs": write_agent_logs,
            "worker": worker,
            "batch_size": batch_size,
            "preview_limit": preview_limit,
            "config": copy.deepcopy(config),
        },
        world_size=world_size,
        samples=all_results,
    )
    log(f"  Results saved: {eval_path}")
    if write_agent_logs:
        log(f"  Agent logs saved: {agent_log_path}")

    if not run_baseline:
        if is_dist:
            dist.barrier()
        cleanup_eval_distributed()
        return

    # ── Also run baseline (no multi-agent, just the model) ──
    log(f"\n{'='*60}")
    log("  BASELINE (single model, no multi-agent)")
    log(f"{'='*60}")

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
                "generation_max_new_tokens": generation_max_new_tokens,
                "inference_mode": inference_mode,
                "use_terminal_prefix": use_terminal_prefix,
                "communication_mode": communication_mode,
                "text_message_edge_threshold": text_message_edge_threshold,
                "text_message_max_new_tokens": text_message_max_new_tokens,
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
            log(f"  [{baseline_total}/{len(full_dataset)}] Baseline Acc: {baseline_acc:.1f}%")

    baseline_acc = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0.0

    log(f"\n{'='*60}")
    log("  COMPARISON")
    log(f"{'='*60}")
    log(f"  Multi-agent (trained): {accuracy:.2f}%")
    log(f"  Single model baseline: {baseline_acc:.2f}%")
    log(f"  Improvement:           {accuracy - baseline_acc:+.2f}%")
    log(f"{'='*60}")

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
            "generation_max_new_tokens": generation_max_new_tokens,
            "inference_mode": inference_mode,
            "use_terminal_prefix": use_terminal_prefix,
            "communication_mode": communication_mode,
            "text_message_edge_threshold": text_message_edge_threshold,
            "text_message_max_new_tokens": text_message_max_new_tokens,
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
        "--communication-mode",
        type=str,
        default="latent_prefix",
        choices=["latent_prefix", "text_messages"],
    )
    parser.add_argument("--text-message-edge-threshold", type=float, default=0.5)
    parser.add_argument("--text-message-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Also run the embedded single-model baseline after ours eval.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-agent-logs", action="store_true")
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--preview-limit", type=int, default=5)
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.max_samples,
        args.split,
        args.max_new_tokens,
        inference_mode=args.inference_mode,
        use_terminal_prefix=not args.no_terminal_prefix,
        communication_mode=args.communication_mode,
        text_message_edge_threshold=args.text_message_edge_threshold,
        text_message_max_new_tokens=args.text_message_max_new_tokens,
        run_baseline=args.run_baseline,
        do_sample=args.do_sample,
        write_agent_logs=not args.no_agent_logs,
        worker=args.worker,
        batch_size=args.batch_size,
        preview_limit=args.preview_limit,
    )
