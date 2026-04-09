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
import inspect
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm.auto import tqdm

from src.utils.config import load_config
from src.utils.answer_extraction import extract_answer, math_is_equivalent, MATH_EQUIVALENT_TASKS
from src.pipeline.multi_agent_system import MultiAgentSystem
from src.data import create_dataset
from src.models.agent import Agent


def merge_eval_config(base_config_path: str | Path, eval_config_path: str | Path | None) -> dict:
    config = load_config(base_config_path)
    if eval_config_path is None:
        return config

    eval_payload = yaml.safe_load(Path(eval_config_path).read_text(encoding="utf-8")) or {}
    eval_task = eval_payload.get("task")
    if eval_task:
        config.setdefault("training", {})["task"] = eval_task

    evaluation_cfg = config.setdefault("evaluation", {})
    for key, value in eval_payload.items():
        if key == "task":
            continue
        evaluation_cfg[key] = value
    return config


def persist_eval_configs(
    *,
    output_dir: str | Path,
    eval_config_path: str | Path | None,
) -> dict[str, str | None]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_eval_config_path = None
    if eval_config_path is not None:
        written_eval_config_path = output_dir / "eval_config.yaml"
        written_eval_config_path.write_text(
            Path(eval_config_path).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    return {
        "eval_config_path": str(written_eval_config_path) if written_eval_config_path is not None else None,
    }


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
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return value[index]
    return value


def _sanitize_for_gather(obj):
    """Convert tensors to Python types so all_gather_object (pickle) works reliably."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_for_gather(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_gather(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_gather(v) for v in obj)
    return obj


def _build_progress_iterator(total_steps: int, *, desc: str, rank: int):
    if not is_main_process(rank):
        return range(total_steps)
    return tqdm(
        range(total_steps),
        total=total_steps,
        desc=desc,
        dynamic_ncols=True,
    )


def _close_progress_iterator(progress) -> None:
    close = getattr(progress, "close", None)
    if callable(close):
        close()


def _generation_hit_token_limit(generation: dict, *, max_new_tokens: int) -> bool:
    finish_reason = generation.get("finish_reason")
    if finish_reason == "max_new_tokens":
        return True

    generated_token_count = generation.get("generated_token_count")
    if generated_token_count is None:
        return False
    return bool(generated_token_count >= max_new_tokens and not generation.get("stopped_early", True))


def _print_token_limit_warning(
    *,
    question_id: str,
    question: str,
    prediction: str,
    generation: dict,
    max_new_tokens: int,
) -> None:
    if not _generation_hit_token_limit(generation, max_new_tokens=max_new_tokens):
        return

    red = "\033[31m"
    reset = "\033[0m"
    generated_text = generation.get("generated_text") or prediction or ""
    preview = generated_text.replace("\n", " ")[:200]
    print(
        f"{red}  Max-token warning [{question_id}]: "
        f"finish_reason={generation.get('finish_reason')}, "
        f"generated_token_count={generation.get('generated_token_count')}, "
        f"limit={max_new_tokens}{reset}"
    )
    print(f"{red}    Q: {question[:120]}{reset}")
    print(f"{red}    A: {preview}{reset}")


def write_eval_snapshot(
    eval_path: Path,
    method: str,
    task: str,
    correct: int,
    total: int,
    time_seconds: float,
    *,
    valid: int | None = None,
    valid_correct: int | None = None,
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
    if valid is None:
        valid = total
    if valid_correct is None:
        valid_correct = correct
    truncated = total - valid
    valid_accuracy = valid_correct / valid * 100 if valid > 0 else 0.0
    payload = {
        "method": method,
        "task": task,
        "metrics": {
            "accuracy": accuracy,
            "valid_accuracy": valid_accuracy,
            "correct": correct,
            "total": total,
            "valid": valid,
            "truncated": truncated,
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


def write_humaneval_snapshot(
    eval_path: Path,
    method: str,
    task: str,
    *,
    metrics: dict,
    parameters: dict,
    world_size: int,
    samples: list[dict],
    artifacts: dict,
) -> None:
    payload = {
        "method": method,
        "task": task,
        "metrics": metrics,
        "parameters": parameters,
        "world_size": world_size,
        "samples": samples,
        "artifacts": artifacts,
    }
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _import_human_eval_evaluation():
    try:
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        return None
    return evaluate_functional_correctness


def _evaluate_humaneval_samples(
    sample_path: Path,
    *,
    num_samples_per_task: int,
    pass_at_k: list[int],
    config: dict,
    problems: list[dict],
    output_dir: Path,
) -> dict:
    evaluator = _import_human_eval_evaluation()
    if evaluator is None:
        raise RuntimeError(
            "HumanEval evaluation requires the `human_eval` package. "
            "Install it and enable its execution harness before running this task."
        )

    problem_path = output_dir / "humaneval_problems.jsonl"
    problem_rows = [
        {
            "task_id": sample["task_id"],
            "prompt": sample["prompt"],
            "canonical_solution": sample["canonical_solution"],
            "test": sample["test"],
            "entry_point": sample["entry_point"],
        }
        for sample in problems
    ]
    _write_jsonl(problem_path, problem_rows)

    eval_kwargs = {}
    evaluator_signature = inspect.signature(evaluator)
    if "problem_file" in evaluator_signature.parameters:
        eval_kwargs["problem_file"] = str(problem_path)
    elif "problem_file_path" in evaluator_signature.parameters:
        eval_kwargs["problem_file_path"] = str(problem_path)
    if "k" in evaluator_signature.parameters:
        eval_kwargs["k"] = list(pass_at_k)
    if "n_workers" in evaluator_signature.parameters:
        eval_kwargs["n_workers"] = int(config.get("evaluation", {}).get("n_workers", 4))
    if "timeout" in evaluator_signature.parameters:
        eval_kwargs["timeout"] = float(config.get("evaluation", {}).get("timeout", 3.0))

    result = evaluator(str(sample_path), **eval_kwargs)
    results_path = Path(f"{sample_path}_results.jsonl")
    return {
        "metrics": result if isinstance(result, dict) else {"pass@1": result},
        "problem_path": str(problem_path),
        "results_path": str(results_path),
        "pass_at_k": list(pass_at_k),
    }


def _evaluate_loaded_system_humaneval(
    system,
    config: dict,
    *,
    config_path: str,
    output_dir: Path,
    checkpoint_path: str | None,
    max_samples: int | None,
    split: str,
    max_new_tokens: int,
    inference_mode: str,
    use_terminal_prefix: bool,
    do_sample: bool,
    write_agent_logs: bool,
    worker: int | None,
    batch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
    is_dist: bool,
    cleanup_distributed: bool,
):
    evaluation_cfg = config.get("evaluation", {})
    num_samples_per_task = int(evaluation_cfg.get("num_samples_per_task", 1))
    pass_at_k = list(evaluation_cfg.get("pass_at_k", [1]))
    if max_samples is not None and max_samples < 0:
        max_samples = None

    if is_main_process(rank):
        print("\nRunning HumanEval evaluation...")

    full_dataset = create_dataset(task="humaneval", split=split, max_samples=max_samples)
    base_problems = [full_dataset[idx] for idx in range(len(full_dataset))]
    repeated_samples: list[dict] = []
    for sample in base_problems:
        repeated_samples.extend([sample] * num_samples_per_task)

    repeated_dataset = shard_dataset(repeated_samples, rank, world_size)
    dataloader = DataLoader(repeated_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sample_rows: list[dict] = []
    agent_log_rows: list[dict] = []
    sample_durations: list[float] = []
    generated_token_counts: list[int] = []
    t_start = time.time()

    local_steps = len(dataloader)
    step_counts = gather_sharded_objects(local_steps, rank, world_size)
    max_steps = max(step_counts)
    dataloader_iter = iter(dataloader)

    progress = _build_progress_iterator(max_steps, desc="HumanEval", rank=rank)

    with torch.no_grad():
        for idx in progress:
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
                    max_new_tokens=max_new_tokens,
                    inference_mode=inference_mode,
                    use_terminal_prefix=use_terminal_prefix,
                    do_sample=do_sample,
                    collect_agent_logs=write_agent_logs,
                )

                generated_text = output["generated_text"]
                generation = output.get("generation", {})
                agent_logs = output.get("agent_logs", [])
                batch_generated_text = generated_text if isinstance(generated_text, list) else [generated_text]
                batch_question_ids = batch["question_ids"]
                batch_questions = batch["questions"]
                batch_answers = batch["answers"]
                batch_size_actual = len(batch_questions)
                batch_elapsed = time.time() - t0
                per_sample_seconds = batch_elapsed / max(batch_size_actual, 1)
                local_update = []
                for sample_idx in range(batch_size_actual):
                    sample_generation = {
                        key: _select_batch_item(value, sample_idx)
                        for key, value in generation.items()
                    }
                    local_update.append(
                        {
                            "task_id": str(batch_question_ids[sample_idx]),
                            "completion": batch_generated_text[sample_idx],
                            "question": batch_questions[sample_idx],
                            "answer": batch_answers[sample_idx],
                            "generation": sample_generation,
                            "sample_seconds": per_sample_seconds,
                            "agent_log": build_agent_sample_log(
                                question_id=batch_question_ids[sample_idx],
                                question=batch_questions[sample_idx],
                                gold=batch_answers[sample_idx],
                                prediction=batch_generated_text[sample_idx],
                                generation=sample_generation,
                                correct=False,
                                agent_logs=agent_logs,
                            ) if write_agent_logs else None,
                        }
                    )

            gathered_updates = gather_sharded_objects(local_update, rank, world_size)

            if not is_main_process(rank):
                continue

            for shard_updates in gathered_updates:
                if shard_updates is None:
                    continue
                for shard_update in shard_updates:
                    sample_durations.append(shard_update["sample_seconds"])
                    generated_token_counts.append(shard_update["generation"].get("generated_token_count", 0))
                    sample_rows.append(
                        {
                            "task_id": shard_update["task_id"],
                            "completion": shard_update["completion"],
                            "question": shard_update["question"],
                        }
                    )
                    if write_agent_logs and shard_update["agent_log"] is not None:
                        agent_log_rows.append(shard_update["agent_log"])
                    _print_token_limit_warning(
                        question_id=shard_update["task_id"],
                        question=shard_update["question"],
                        prediction=shard_update["completion"],
                        generation=shard_update["generation"],
                        max_new_tokens=max_new_tokens,
                    )
    _close_progress_iterator(progress)

    all_rows = sample_rows
    all_agent_log_rows = agent_log_rows

    if not is_main_process(rank):
        if is_dist:
            dist.barrier()
            if cleanup_distributed:
                cleanup_eval_distributed()
        return None

    samples_path = output_dir / "humaneval_samples.jsonl"
    _write_jsonl(
        samples_path,
        [{"task_id": row["task_id"], "completion": extract_answer(row["completion"], task_type="humaneval")} for row in all_rows],
    )

    harness_result = _evaluate_humaneval_samples(
        samples_path,
        num_samples_per_task=num_samples_per_task,
        pass_at_k=pass_at_k,
        config=config,
        problems=base_problems,
        output_dir=output_dir,
    )

    # Merge harness pass/fail results back into samples
    results_path = Path(harness_result["results_path"])
    if results_path.exists():
        harness_by_task = {}
        with results_path.open() as f:
            for line in f:
                row = json.loads(line)
                harness_by_task[row["task_id"]] = {
                    "passed": row.get("passed", False),
                    "result": row.get("result", ""),
                }
        for row in all_rows:
            hr = harness_by_task.get(row["task_id"], {})
            row["passed"] = hr.get("passed", False)
            row["result"] = hr.get("result", "")

    eval_path = output_dir / "eval_results.json"
    agent_log_path = output_dir / "agent_logs.json"
    role_agent_log_dir = output_dir / "agent_log"
    metrics = {
        **harness_result["metrics"],
        "time_seconds": time.time() - t_start,
        "num_tasks": len(base_problems),
        "num_samples_per_task": num_samples_per_task,
        "total_samples": len(all_rows),
        "pass_at_k": pass_at_k,
        "avg_sample_seconds": (sum(sample_durations) / len(sample_durations)) if sample_durations else None,
        "avg_generated_tokens": (
            sum(generated_token_counts) / len(generated_token_counts)
        ) if generated_token_counts else None,
        "avg_tokens_per_second": (
            (sum(generated_token_counts) / sum(sample_durations))
            if sample_durations and sum(sample_durations) > 0
            else None
        ),
    }
    parameters = {
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "split": split,
        "max_samples": max_samples,
        "generation_max_new_tokens": max_new_tokens,
        "inference_mode": inference_mode,
        "use_terminal_prefix": use_terminal_prefix,
        "do_sample": do_sample,
        "write_agent_logs": write_agent_logs,
        "worker": worker,
        "batch_size": batch_size,
        "config": copy.deepcopy(config),
    }
    artifacts = {
        "eval_path": str(eval_path),
        "samples_path": str(samples_path),
        "problem_path": harness_result["problem_path"],
        "results_path": harness_result["results_path"],
    }
    write_humaneval_snapshot(
        eval_path=eval_path,
        method="ours_trained_multi_agent",
        task="humaneval",
        metrics=metrics,
        parameters=parameters,
        world_size=world_size,
        samples=all_rows,
        artifacts=artifacts,
    )
    if write_agent_logs:
        write_agent_log_snapshot(
            agent_log_path=agent_log_path,
            method="ours_trained_multi_agent",
            task="humaneval",
            parameters=parameters,
            samples=all_agent_log_rows,
        )
        write_role_agent_log_snapshots(role_agent_log_dir, all_agent_log_rows)

    result_payload = {
        "method": "ours_trained_multi_agent",
        "task": "humaneval",
        "metrics": metrics,
        "paths": {
            **artifacts,
            "agent_log_path": str(agent_log_path) if write_agent_logs else None,
            "role_agent_log_dir": str(role_agent_log_dir) if write_agent_logs else None,
        },
    }
    print(f"  HumanEval samples saved: {samples_path}")
    print(f"  HumanEval results saved: {harness_result['results_path']}")

    if is_dist:
        dist.barrier()
    if cleanup_distributed:
        cleanup_eval_distributed()
    return result_payload


def setup_eval_distributed() -> tuple[torch.device, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
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


def file_based_gather(local_obj, rank: int, world_size: int, output_dir: Path):
    """Gather objects via temp files — works regardless of backend."""
    if world_size <= 1:
        return [local_obj]
    shard_dir = output_dir / ".gather_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank_{rank}.pkl"
    with open(shard_path, "wb") as f:
        pickle.dump(local_obj, f)
    # Wait until all ranks have written
    dist.barrier()
    # Rank 0 reads all shards
    gathered = []
    if rank == 0:
        for r in range(world_size):
            p = shard_dir / f"rank_{r}.pkl"
            with open(p, "rb") as f:
                gathered.append(pickle.load(f))
    # Wait for rank 0 to finish reading before anyone cleans up
    dist.barrier()
    # Clean up own shard
    if shard_path.exists():
        shard_path.unlink()
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
    cleanup_distributed: bool = True,
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
    if task == "humaneval":
        return _evaluate_loaded_system_humaneval(
            system=system,
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            max_samples=max_samples,
            split=split,
            max_new_tokens=generation_max_new_tokens,
            inference_mode=inference_mode,
            use_terminal_prefix=use_terminal_prefix,
            do_sample=do_sample,
            write_agent_logs=write_agent_logs,
            worker=worker,
            batch_size=batch_size,
            device=device,
            rank=rank,
            world_size=world_size,
            is_dist=is_dist,
            cleanup_distributed=cleanup_distributed,
        )
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
    valid = 0
    valid_correct = 0
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

    # Each worker writes results to its own shard file (no inter-worker sync during loop)
    shard_dir = output_dir / ".eval_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank_{rank}.pkl"
    if shard_path.exists():
        shard_path.unlink()

    if is_main_process(rank):
        progress_bar = tqdm(
            total=len(full_dataset),
            desc=f"Eval {task}",
            dynamic_ncols=True,
        )
    else:
        progress_bar = None

    dataloader_iter = iter(dataloader)
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

    local_results = []
    with torch.no_grad():
        for batch in dataloader_iter:
            t0 = time.time()
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

            for sample_idx in range(batch_size_actual):
                sample_generation = {
                    key: _select_batch_item(value, sample_idx)
                    for key, value in generation.items()
                }
                sample_generation = _sanitize_for_gather(sample_generation)
                generated_count = sample_generation.get("generated_token_count", 0)
                truncated = generated_count >= generation_max_new_tokens
                pred = extract_answer(batch_generated_text[sample_idx], task_type=task)
                gold = batch_answers[sample_idx] if task in MATH_EQUIVALENT_TASKS else extract_answer(batch_answers[sample_idx], task_type=task)
                is_correct = math_is_equivalent(pred, gold) if task in MATH_EQUIVALENT_TASKS else pred.strip() == gold.strip()
                agent_log_entry = None
                if write_agent_logs:
                    agent_log_entry = _sanitize_for_gather(build_agent_sample_log(
                        question_id=batch_question_ids[sample_idx],
                        question=batch_questions[sample_idx],
                        gold=gold,
                        prediction=pred,
                        generation=sample_generation,
                        correct=is_correct,
                        agent_logs=agent_logs,
                    ))
                local_results.append({
                    "question": batch_questions[sample_idx],
                    "question_id": batch_question_ids[sample_idx],
                    "gold": gold,
                    "prediction": pred,
                    "generation": sample_generation,
                    "correct": is_correct,
                    "truncated": truncated,
                    "agent_log": agent_log_entry,
                    "sample_seconds": per_sample_seconds,
                })

            # Write this worker's cumulative results to shard file
            with open(shard_path, "wb") as f:
                pickle.dump(local_results, f)

            # Rank 0: merge all available shards and update JSON
            if is_main_process(rank):
                correct = 0
                total = 0
                valid = 0
                valid_correct = 0
                results = []
                agent_log_results = []
                sample_durations = []
                generated_token_counts = []
                for r in range(world_size):
                    rp = shard_dir / f"rank_{r}.pkl"
                    if not rp.exists():
                        continue
                    try:
                        with open(rp, "rb") as f:
                            shard_items = pickle.load(f)
                    except (EOFError, pickle.UnpicklingError):
                        continue
                    for item in shard_items:
                        total += 1
                        if item["correct"]:
                            correct += 1
                        if not item.get("truncated", False):
                            valid += 1
                            if item["correct"]:
                                valid_correct += 1
                        sample_durations.append(item["sample_seconds"])
                        generated_token_counts.append(item["generation"].get("generated_token_count", 0))
                        results.append({
                            "question": item["question"],
                            "question_id": item["question_id"],
                            "gold": item["gold"],
                            "prediction": item["prediction"],
                            "generation": item["generation"],
                            "correct": item["correct"],
                            "truncated": item.get("truncated", False),
                        })
                        if write_agent_logs and item.get("agent_log") is not None:
                            agent_log_results.append(item["agent_log"])

                if progress_bar is not None and total > 0:
                    acc = correct / total * 100
                    progress_bar.n = total
                    progress_bar.set_postfix(acc=f"{acc:.1f}%", correct=f"{correct}/{total}")
                    progress_bar.refresh()

                write_eval_snapshot(
                    eval_path=eval_path,
                    method="ours_trained_multi_agent",
                    task=task,
                    correct=correct,
                    total=total,
                    valid=valid,
                    valid_correct=valid_correct,
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

    # Wait for all workers to finish
    if is_dist:
        dist.barrier()

    # Final merge on rank 0
    if is_main_process(rank):
        correct = 0
        total = 0
        valid = 0
        valid_correct = 0
        results = []
        agent_log_results = []
        sample_durations = []
        generated_token_counts = []
        for r in range(world_size):
            rp = shard_dir / f"rank_{r}.pkl"
            if not rp.exists():
                continue
            with open(rp, "rb") as f:
                shard_items = pickle.load(f)
            for item in shard_items:
                total += 1
                if item["correct"]:
                    correct += 1
                if not item.get("truncated", False):
                    valid += 1
                    if item["correct"]:
                        valid_correct += 1
                sample_durations.append(item["sample_seconds"])
                generated_token_counts.append(item["generation"].get("generated_token_count", 0))
                results.append({
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "gold": item["gold"],
                    "prediction": item["prediction"],
                    "generation": item["generation"],
                    "correct": item["correct"],
                    "truncated": item.get("truncated", False),
                })
                if write_agent_logs and item.get("agent_log") is not None:
                    agent_log_results.append(item["agent_log"])

        write_eval_snapshot(
            eval_path=eval_path,
            method="ours_trained_multi_agent",
            task=task,
            correct=correct,
            total=total,
            valid=valid,
            valid_correct=valid_correct,
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

    if progress_bar is not None:
        progress_bar.close()

    # Clean up shard files
    if is_main_process(rank):
        if shard_dir.exists():
            shutil.rmtree(shard_dir)

    if not is_main_process(rank):
        if cleanup_distributed and dist.is_initialized():
            dist.destroy_process_group()
        return None

    t_total = time.time() - t_start
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Task:     {task}")
    print(f"  Split:    {split}")
    truncated_count = total - valid
    valid_accuracy = valid_correct / valid * 100 if valid > 0 else 0.0
    print(f"  Samples:  {total}")
    print(f"  Valid:    {valid} ({valid/total*100:.1f}%)" if total > 0 else f"  Valid:    0")
    print(f"  Truncated:{truncated_count}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}% (over all samples)")
    print(f"  Valid Acc:{valid_accuracy:.2f}% (over valid samples only)")
    print(f"  Time:     {t_total:.1f}s ({t_total/max(total, 1):.1f}s/sample)")
    if sample_durations:
        print(f"  Avg sample: {sum(sample_durations)/len(sample_durations):.2f}s")
    if generated_token_counts:
        print(f"  Avg generated tokens: {sum(generated_token_counts)/len(generated_token_counts):.2f}")
        if sample_durations and sum(sample_durations) > 0:
            print(f"  Tokens/sec: {sum(generated_token_counts)/sum(sample_durations):.2f}")
    print(f"{'='*60}")

    result_payload = {
        "method": "ours_trained_multi_agent",
        "task": task,
        "metrics": {
            "accuracy": accuracy,
            "valid_accuracy": valid_accuracy,
            "correct": correct,
            "total": total,
            "valid": valid,
            "truncated": total - valid,
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
    if not run_baseline:
        return result_payload

    print(f"\n{'='*60}")
    print(f"  BASELINE (single model, no multi-agent)")
    print(f"{'='*60}")

    baseline_correct = 0
    baseline_total = 0
    baseline_samples = []

    dataloader_iter = iter(dataloader)
    progress = _build_progress_iterator(max_steps, desc="Baseline", rank=rank)
    for idx in progress:
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
            batch_size_actual = task_ids.shape[0]
            baseline_update = []
            for sample_idx in range(batch_size_actual):
                generated_token_ids = sequences[sample_idx][task_ids.shape[1]:].tolist()
                baseline_text = system.base_model.tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=True,
                )
                generation = build_generation_metadata(
                    generated_token_ids=generated_token_ids,
                    eos_token_id=system.base_model.tokenizer.eos_token_id,
                    max_new_tokens=generation_max_new_tokens,
                )
                generated_count = generation.get("generated_token_count", 0)
                truncated = generated_count >= generation_max_new_tokens
                pred = extract_answer(baseline_text, task_type=task)
                gold = batch["answers"][sample_idx] if task in MATH_EQUIVALENT_TASKS else extract_answer(batch["answers"][sample_idx], task_type=task)
                is_correct = math_is_equivalent(pred, gold) if task in MATH_EQUIVALENT_TASKS else pred.strip() == gold.strip()
                baseline_update.append(
                    {
                        "question_id": batch["question_ids"][sample_idx],
                        "question": batch["questions"][sample_idx],
                        "gold": gold,
                        "prediction": pred,
                        "generation": generation,
                        "correct": is_correct,
                        "truncated": truncated,
                    }
                )

        gathered_baseline_updates = gather_sharded_objects(baseline_update, rank, world_size)
        if not is_main_process(rank):
            continue
        for shard_updates in gathered_baseline_updates:
            if shard_updates is None:
                continue
            for shard_update in shard_updates:
                if shard_update["correct"]:
                    baseline_correct += 1
                baseline_total += 1
                baseline_samples.append(shard_update)
                _print_token_limit_warning(
                    question_id=str(shard_update["question_id"]),
                    question=shard_update["question"],
                    prediction=shard_update["prediction"],
                    generation=shard_update["generation"],
                    max_new_tokens=generation_max_new_tokens,
                )

        baseline_acc = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0.0
        write_eval_snapshot(
            eval_path=eval_path,
            method="ours_trained_multi_agent",
            task=task,
            correct=correct,
            total=total,
            time_seconds=t_total,
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

    _close_progress_iterator(progress)
    baseline_acc = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Multi-agent (trained): {accuracy:.2f}%")
    print(f"  Single model baseline: {baseline_acc:.2f}%")
    print(f"  Improvement:           {accuracy - baseline_acc:+.2f}%")
    print(f"{'='*60}")

    write_eval_snapshot(
        eval_path=eval_path,
        method="ours_trained_multi_agent",
        task=task,
        correct=correct,
        total=total,
        time_seconds=t_total,
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
    result_payload["baseline_single_model"] = {
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
    result_payload["comparison"] = {
        "baseline_accuracy": baseline_acc,
        "improvement": accuracy - baseline_acc,
    }
    return result_payload


def _build_and_load_system(config, checkpoint_path, device):
    """Build MultiAgentSystem and load checkpoint onto device."""
    system = MultiAgentSystem(config)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        base_model_state = ckpt.get("base_model_state")
        if base_model_state is not None:
            system.base_model.model.load_state_dict(base_model_state)
        comp_state = ckpt["compressor_state"]
        if isinstance(comp_state, list):
            for i, state in enumerate(comp_state):
                cleaned = {k.replace("module.", "") if k.startswith("module.") else k: v for k, v in state.items()}
                system.compressors[i].load_state_dict(cleaned)
        else:
            cleaned_state = {}
            for k, v in comp_state.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                cleaned_state[new_key] = v
            system.compressor.load_state_dict(cleaned_state)
        system.adjacency.load_state_dict(ckpt["adjacency_state"])
    system.to(device)
    system.eval()
    return system


def _gpu_worker(
    gpu_id: int,
    dataset_items: list[dict],
    config: dict,
    checkpoint_path: str | None,
    task: str,
    generation_max_new_tokens: int,
    inference_mode: str,
    use_terminal_prefix: bool,
    do_sample: bool,
    write_agent_logs: bool,
    batch_size: int,
    result_queue,
):
    """Independent single-GPU eval worker. No distributed."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    system = _build_and_load_system(config, checkpoint_path, device)

    dataloader = DataLoader(dataset_items, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in dataloader:
            t0 = time.time()
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

            for sample_idx in range(batch_size_actual):
                sample_generation = {
                    key: _select_batch_item(value, sample_idx)
                    for key, value in generation.items()
                }
                sample_generation = _sanitize_for_gather(sample_generation)
                generated_count = sample_generation.get("generated_token_count", 0)
                truncated = generated_count >= generation_max_new_tokens
                pred = extract_answer(batch_generated_text[sample_idx], task_type=task)
                gold = batch_answers[sample_idx] if task in MATH_EQUIVALENT_TASKS else extract_answer(batch_answers[sample_idx], task_type=task)
                is_correct = math_is_equivalent(pred, gold) if task in MATH_EQUIVALENT_TASKS else pred.strip() == gold.strip()
                agent_log_entry = None
                if write_agent_logs:
                    agent_log_entry = _sanitize_for_gather(build_agent_sample_log(
                        question_id=batch_question_ids[sample_idx],
                        question=batch_questions[sample_idx],
                        gold=gold,
                        prediction=pred,
                        generation=sample_generation,
                        correct=is_correct,
                        agent_logs=agent_logs,
                    ))
                result_queue.put({
                    "question": batch_questions[sample_idx],
                    "question_id": batch_question_ids[sample_idx],
                    "gold": gold,
                    "prediction": pred,
                    "generation": sample_generation,
                    "correct": is_correct,
                    "truncated": truncated,
                    "agent_log": agent_log_entry,
                    "sample_seconds": per_sample_seconds,
                    "gpu_id": gpu_id,
                })

    result_queue.put(None)  # signal done

    # Explicit cleanup to ensure process exits
    del system
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def evaluate(
    config_path: str,
    checkpoint_path: str | None = None,
    max_samples: int | None = None,
    split: str | None = None,
    max_new_tokens: int | None = None,
    inference_mode: str | None = None,
    use_terminal_prefix: bool | None = None,
    run_baseline: bool = False,
    do_sample: bool | None = None,
    write_agent_logs: bool | None = None,
    worker: int | None = None,
    batch_size: int | None = None,
    question: str | None = None,
    output_dir: str | Path | None = None,
    eval_config_path: str | None = None,
    model_dtype: str | None = None,
    num_gpus: int | None = None,
):
    import torch.multiprocessing as mp

    config = merge_eval_config(config_path, eval_config_path)
    evaluation_cfg = config.get("evaluation", {})
    split = split or evaluation_cfg.get("split", "test")
    generation_max_new_tokens = max_new_tokens or evaluation_cfg.get("max_new_tokens", 4096)
    inference_mode = inference_mode or evaluation_cfg.get("inference_mode", "chat_with_prefix")
    if use_terminal_prefix is None:
        use_terminal_prefix = bool(evaluation_cfg.get("use_terminal_prefix", True))
    if do_sample is None:
        do_sample = bool(evaluation_cfg.get("do_sample", False))
    if write_agent_logs is None:
        write_agent_logs = bool(evaluation_cfg.get("write_agent_logs", True))
    if worker is None:
        worker = evaluation_cfg.get("worker")
    if batch_size is None:
        batch_size = int(evaluation_cfg.get("batch_size", 1))
    if max_samples is None:
        max_samples = evaluation_cfg.get("max_samples")
    if model_dtype is not None:
        config.setdefault("model", {})["dtype"] = model_dtype

    task = config["training"]["task"]
    checkpoint_dir = Path(output_dir) if output_dir is not None else Path(checkpoint_path).parent
    persist_eval_configs(output_dir=checkpoint_dir, eval_config_path=eval_config_path)

    if max_samples is not None and max_samples < 0:
        max_samples = None

    if question is not None:
        full_dataset = build_single_question_dataset(question)
    else:
        print(f"Loading {task} {split} set...")
        full_dataset = create_dataset(task=task, split=split, max_samples=max_samples)

    # Determine GPU count
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if num_gpus is None:
        num_gpus = available_gpus
    num_gpus = min(num_gpus, available_gpus, len(full_dataset))

    print(f"Total samples: {len(full_dataset)}")
    print(f"GPUs: {num_gpus}")
    print(f"Batch size per GPU: {batch_size}")

    # Show adjacency on GPU 0
    device0 = torch.device("cuda:0")
    system0 = _build_and_load_system(config, checkpoint_path, device0)
    print(f"\nLearned adjacency:")
    print(system0.log_adjacency())
    A = system0.adjacency.get_adjacency().detach()
    print(f"Range: [{A.min().item():.4f}, {A.max().item():.4f}]")
    print(f"Hard adjacency (threshold=0.5):")
    print(system0.adjacency.get_hard_adjacency())
    del system0
    torch.cuda.empty_cache()

    # Split dataset into per-GPU shards
    shards = []
    for i in range(num_gpus):
        indices = list(range(i, len(full_dataset), num_gpus))
        shard_items = [full_dataset[j] for j in indices]
        shards.append(shard_items)
        print(f"  GPU {i}: {len(shard_items)} samples")

    # Setup output paths
    eval_stem = "eval_results" if split == "test" else f"eval_results_{split}"
    agent_log_stem = "agent_logs" if split == "test" else f"agent_logs_{split}"
    eval_path = checkpoint_dir / f"{eval_stem}.json"
    agent_log_path = checkpoint_dir / f"{agent_log_stem}.json"
    role_agent_log_dir = checkpoint_dir / "agent_log"
    for path in (eval_path, agent_log_path):
        if path.exists():
            path.unlink()
    if role_agent_log_dir.exists():
        shutil.rmtree(role_agent_log_dir)

    params = {
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "eval_config_path": eval_config_path,
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
        "num_gpus": num_gpus,
        "config": copy.deepcopy(config),
    }

    # Spawn workers
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    print(f"\nSpawning {num_gpus} GPU workers...")
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=_gpu_worker,
            args=(
                gpu_id, shards[gpu_id], config, checkpoint_path, task,
                generation_max_new_tokens, inference_mode, use_terminal_prefix,
                do_sample, write_agent_logs, batch_size, result_queue,
            ),
        )
        p.start()
        workers.append(p)

    # Collect results from all workers
    print("Running evaluation...")
    t_start = time.time()
    progress = tqdm(total=len(full_dataset), desc=f"Eval {task}", dynamic_ncols=True)

    correct = 0
    total = 0
    valid = 0
    valid_correct = 0
    results = []
    agent_log_results = []
    sample_durations = []
    generated_token_counts = []
    done_count = 0

    while done_count < num_gpus:
        item = result_queue.get()
        if item is None:
            done_count += 1
            continue

        total += 1
        if item["correct"]:
            correct += 1
        if not item.get("truncated", False):
            valid += 1
            if item["correct"]:
                valid_correct += 1
        sample_durations.append(item["sample_seconds"])
        generated_token_counts.append(item["generation"].get("generated_token_count", 0))
        results.append({
            "question": item["question"],
            "question_id": item["question_id"],
            "gold": item["gold"],
            "prediction": item["prediction"],
            "generation": item["generation"],
            "correct": item["correct"],
            "truncated": item.get("truncated", False),
        })
        if write_agent_logs and item.get("agent_log") is not None:
            agent_log_results.append(item["agent_log"])

        # Update progress with accuracy
        acc = correct / total * 100
        progress.update(1)
        progress.set_postfix(acc=f"{acc:.1f}%", correct=f"{correct}/{total}")

        # Update JSON after every datapoint
        write_eval_snapshot(
            eval_path=eval_path,
            method="ours_trained_multi_agent",
            task=task,
            correct=correct,
            total=total,
            valid=valid,
            valid_correct=valid_correct,
            time_seconds=time.time() - t_start,
            avg_sample_seconds=(sum(sample_durations) / len(sample_durations)) if sample_durations else None,
            avg_generated_tokens=(sum(generated_token_counts) / len(generated_token_counts)) if generated_token_counts else None,
            avg_tokens_per_second=(
                (sum(generated_token_counts) / sum(sample_durations))
                if sample_durations and sum(sample_durations) > 0
                else None
            ),
            parameters=params,
            world_size=num_gpus,
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

    progress.close()

    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)
        if p.is_alive():
            p.kill()

    # ── Summary ──
    t_total = time.time() - t_start
    accuracy = correct / total * 100 if total > 0 else 0.0
    valid_accuracy = valid_correct / valid * 100 if valid > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Task:     {task}")
    print(f"  Split:    {split}")
    print(f"  GPUs:     {num_gpus}")
    print(f"  Samples:  {total}")
    print(f"  Valid:    {valid} ({valid/total*100:.1f}%)" if total > 0 else f"  Valid:    0")
    print(f"  Truncated:{total - valid}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}% (over all samples)")
    print(f"  Valid Acc:{valid_accuracy:.2f}% (over valid samples only)")
    print(f"  Time:     {t_total:.1f}s ({t_total/max(total, 1):.1f}s/sample)")
    if sample_durations:
        print(f"  Avg sample: {sum(sample_durations)/len(sample_durations):.2f}s")
    if generated_token_counts:
        avg_tokens = sum(generated_token_counts) / len(generated_token_counts)
        print(f"  Avg gen tokens: {avg_tokens:.1f}")
        if sum(sample_durations) > 0:
            print(f"  Tokens/s: {sum(generated_token_counts)/sum(sample_durations):.2f}")
    print(f"{'='*60}")
    print(f"  Results saved: {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "test"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--inference-mode",
        type=str,
        default=None,
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
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16", "float16"],
                        help="Override model dtype for eval (default: use training config)")
    parser.add_argument("--no-agent-logs", action="store_true")
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (default: all available)")
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.max_samples,
        args.split,
        args.max_new_tokens,
        inference_mode=args.inference_mode,
        use_terminal_prefix=(False if args.no_terminal_prefix else None),
        run_baseline=args.run_baseline,
        do_sample=(True if args.do_sample else None),
        write_agent_logs=(False if args.no_agent_logs else None),
        worker=args.worker,
        batch_size=args.batch_size,
        question=args.question,
        output_dir=args.output_dir,
        eval_config_path=args.eval_config,
        model_dtype=args.dtype,
        num_gpus=args.num_gpus,
    )
