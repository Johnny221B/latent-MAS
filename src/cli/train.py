"""
Unified training entry point with proper multi-GPU DDP support.

Single GPU:
    uv run --python .venv/bin/python torchrun --nproc_per_node=1 src/cli/train.py --config configs/experiments/gsm8k_3agent.yaml

Multi-GPU (8 cards):
    uv run --python .venv/bin/python torchrun --nproc_per_node=8 src/cli/train.py --config configs/experiments/gsm8k_3agent.yaml

Specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --python .venv/bin/python torchrun --nproc_per_node=4 src/cli/train.py --config configs/experiments/gsm8k_3agent.yaml
"""

import argparse
import copy
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import csv
from datetime import UTC, datetime
import json
import random

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.optim import AdamW

from src.utils.config import load_config
from src.utils.output_paths import build_timestamped_output_dir
from src.utils.reporting import finish_wandb, init_wandb_run, log_wandb
from src.utils.answer_extraction import extract_answer
from src.utils.token_utils import append_eos_token
from src.utils.training import (
    build_ddp_kwargs,
    compute_grad_norm,
    should_save_checkpoint,
    validate_min_samples_for_batches,
)
from src.pipeline.multi_agent_system import MultiAgentSystem
from src.cli.evaluate import evaluate_loaded_system, gather_sharded_objects, is_main_process as is_eval_main_process, shard_dataset
from src.data import create_dataset


def collate_fn(batch: list[dict]) -> dict:
    payload = {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }
    if batch and "question_id" in batch[0]:
        payload["question_ids"] = [item["question_id"] for item in batch]
    if batch and "raw_answer" in batch[0]:
        payload["raw_answers"] = [item["raw_answer"] for item in batch]
    return payload


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed():
    """Initialize DDP if launched with torchrun or manual launcher."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"DDP initialized: {world_size} processes")
            print(f"  Each process sees {torch.cuda.device_count()} GPU(s)")
        return device, True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Single process mode: {device}")
        return device, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_git_command(args: list[str], *, repo_root: Path | None = None) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root or project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]


def collect_git_provenance(repo_root: Path | None = None) -> dict:
    repo_root = repo_root or project_root
    commit = _run_git_command(["rev-parse", "HEAD"], repo_root=repo_root)
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], repo_root=repo_root)
    status_short = _run_git_command(["status", "--short"], repo_root=repo_root)
    diff_stat = _run_git_command(["diff", "--stat", "HEAD"], repo_root=repo_root)
    return {
        "commit": commit[0] if commit else None,
        "branch": branch[0] if branch else None,
        "status_short": status_short,
        "diff_stat": diff_stat,
        "is_dirty": bool(status_short),
    }


def build_run_provenance(
    *,
    config_path: str,
    output_dir: str | Path,
    training_seed: int,
    world_size: int,
    rank: int,
    is_ddp: bool,
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    runtime_env = env or os.environ
    captured_env = {
        key: runtime_env[key]
        for key in [
            "CUDA_VISIBLE_DEVICES",
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ]
        if key in runtime_env
    }
    return {
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "training": {
            "seed": int(training_seed),
            "is_ddp": bool(is_ddp),
        },
        "launch": {
            "argv": list(argv) if argv is not None else list(sys.argv),
            "rank": int(rank),
            "world_size": int(world_size),
            "environment": captured_env,
        },
        "git": collect_git_provenance(),
    }


def summarize_probe_generation_health(
    samples: list[dict],
    *,
    max_new_tokens: int,
    degeneracy_ratio: float,
) -> dict:
    total = len(samples)
    max_token_hits = 0
    for sample in samples:
        generation = sample.get("generation", {})
        finish_reason = generation.get("finish_reason")
        generated_token_count = generation.get("generated_token_count")
        if finish_reason == "max_new_tokens":
            max_token_hits += 1
            continue
        if generated_token_count is not None and int(generated_token_count) >= int(max_new_tokens):
            max_token_hits += 1
    ratio = (max_token_hits / total) if total > 0 else 0.0
    return {
        "max_new_tokens_count": max_token_hits,
        "max_new_tokens_ratio": ratio,
        "degenerate": total > 0 and ratio >= float(degeneracy_ratio),
    }


def resolve_post_train_eval_plan(evaluation_cfg: dict) -> list[tuple[str, int | None]]:
    split_names = evaluation_cfg.get("splits_after_train", ["train", "test"])
    eval_plan = []
    for split_name in split_names:
        sample_limit = evaluation_cfg.get(f"{split_name}_probe_samples")
        eval_plan.append((split_name, sample_limit))
    return eval_plan


def split_training_and_probe_subsets(
    dataset,
    *,
    probe_samples: int,
    seed: int,
):
    total = len(dataset)
    if probe_samples <= 0:
        return dataset, None, []
    if total <= probe_samples:
        raise ValueError(
            f"Probe split requires more samples than available: total={total}, probe_samples={probe_samples}"
        )

    shuffled_indices = list(range(total))
    random.Random(seed).shuffle(shuffled_indices)
    probe_indices = sorted(shuffled_indices[:probe_samples])
    train_indices = sorted(shuffled_indices[probe_samples:])
    return (
        Subset(dataset, train_indices),
        Subset(dataset, probe_indices),
        probe_indices,
    )


def should_run_training_probe(global_step: int, probe_cfg: dict) -> bool:
    interval = int(probe_cfg.get("every_n_steps", 0))
    return bool(probe_cfg.get("enabled")) and interval > 0 and global_step > 0 and global_step % interval == 0


def evaluate_training_probe(
    *,
    system,
    dataset,
    config: dict,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int,
    max_new_tokens: int,
    inference_mode: str,
    use_terminal_prefix: bool,
    write_agent_logs: bool,
    degeneracy_ratio: float,
):
    if dataset is None:
        return None

    sharded_dataset = shard_dataset(dataset, rank, world_size)
    dataloader = DataLoader(sharded_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    results = []

    system_was_training = getattr(system, "training", False)
    system.eval()
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
                max_new_tokens=max_new_tokens,
                inference_mode=inference_mode,
                use_terminal_prefix=use_terminal_prefix,
                do_sample=False,
                collect_agent_logs=write_agent_logs,
            )
            generated_text = output["generated_text"]
            generation = output.get("generation", {})
            batch_generated_text = generated_text if isinstance(generated_text, list) else [generated_text]
            batch_elapsed = time.time() - t0
            per_sample_seconds = batch_elapsed / max(len(batch["questions"]), 1)

            for sample_idx, question in enumerate(batch["questions"]):
                prediction = extract_answer(batch_generated_text[sample_idx], task_type=config["training"]["task"])
                gold = batch["answers"][sample_idx].strip()
                is_correct = prediction.strip() == gold
                sample_generation = {
                    key: value[sample_idx] if isinstance(value, list) else value
                    for key, value in generation.items()
                }
                results.append(
                    {
                        "question_id": batch["question_ids"][sample_idx],
                        "question": question,
                        "gold": gold,
                        "prediction": prediction,
                        "correct": is_correct,
                        "sample_seconds": per_sample_seconds,
                        "generation": sample_generation,
                    }
                )
    gathered_results = gather_sharded_objects(results, rank, world_size)
    if system_was_training:
        system.train()

    if not is_eval_main_process(rank):
        return None

    merged_results = []
    for shard_results in gathered_results:
        if shard_results is None:
            continue
        merged_results.extend(shard_results)
    merged_results.sort(key=lambda item: str(item["question_id"]))

    correct = sum(int(sample["correct"]) for sample in merged_results)
    total = len(merged_results)
    sample_durations = [float(sample.get("sample_seconds", 0.0)) for sample in merged_results]
    generated_token_counts = [
        float(sample.get("generation", {}).get("generated_token_count", 0))
        for sample in merged_results
    ]
    accuracy = correct / total * 100 if total > 0 else 0.0
    generation_health = summarize_probe_generation_health(
        merged_results,
        max_new_tokens=max_new_tokens,
        degeneracy_ratio=degeneracy_ratio,
    )
    return {
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "time_seconds": sum(sample_durations),
            "avg_sample_seconds": (sum(sample_durations) / len(sample_durations)) if sample_durations else None,
            "avg_generated_tokens": (
                sum(generated_token_counts) / len(generated_token_counts)
            ) if generated_token_counts else None,
            **generation_health,
        },
        "samples": merged_results,
    }


def write_probe_history_json(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def resolve_resume_checkpoint(training_cfg: dict) -> Path | None:
    raw_path = training_cfg.get("resume_from_checkpoint")
    if raw_path is None:
        return None

    checkpoint_path = Path(raw_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_training_output_dir(config: dict) -> Path:
    resume_checkpoint = resolve_resume_checkpoint(config.get("training", {}))
    if resume_checkpoint is not None:
        return resume_checkpoint.parent

    base_output_dir = config.get("output", {}).get("dir", "outputs/run")
    return build_timestamped_output_dir(base_output_dir)


def compute_resume_schedule(
    *,
    checkpoint: dict,
    batches_per_epoch: int,
    grad_accum_steps: int,
) -> dict:
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive when computing a resume schedule")

    global_step = int(checkpoint.get("step", 0))
    if "epoch" in checkpoint and "next_batch_idx" in checkpoint:
        start_epoch = max(int(checkpoint["epoch"]), 0)
        skip_batches_in_epoch = max(int(checkpoint["next_batch_idx"]), 0)
    else:
        completed_batches = global_step * max(int(grad_accum_steps), 1)
        start_epoch = completed_batches // batches_per_epoch
        skip_batches_in_epoch = completed_batches % batches_per_epoch

    start_epoch += skip_batches_in_epoch // batches_per_epoch
    skip_batches_in_epoch = skip_batches_in_epoch % batches_per_epoch
    return {
        "global_step": global_step,
        "start_epoch": start_epoch,
        "skip_batches_in_epoch": skip_batches_in_epoch,
    }


def load_training_checkpoint(
    *,
    checkpoint_path: str | Path,
    compressor,
    adjacency,
    optimizer,
    prefix_projector=None,
    prefix_projectors=None,
    batches_per_epoch: int | None = None,
    grad_accum_steps: int = 1,
    base_model_module=None,
    trainable_base_model: bool = False,
    learnable_prefix_module=None,
) -> dict:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    checkpoint_step = int(checkpoint.get("step", 0))
    if checkpoint_step < 0:
        raise ValueError(f"Checkpoint step must be non-negative: {checkpoint_step}")

    compressor_state = checkpoint.get("compressor_state")
    adjacency_state = checkpoint.get("adjacency_state")
    optimizer_state = checkpoint.get("optimizer_state")
    if adjacency_state is None or optimizer_state is None:
        raise ValueError(
            "Resume checkpoint is missing one of the required training states: "
            "adjacency_state, optimizer_state"
        )
    if compressor_state is None and compressor is not None:
        raise ValueError("Resume checkpoint is missing compressor_state but compressor exists")

    if compressor is not None and compressor_state is not None:
        compressor.load_state_dict(compressor_state)
    adjacency.load_state_dict(adjacency_state)
    optimizer.load_state_dict(optimizer_state)

    learnable_prefix_state = checkpoint.get("learnable_prefix_state")
    if learnable_prefix_module is not None and learnable_prefix_state is not None:
        learnable_prefix_module.load_state_dict(learnable_prefix_state)

    prefix_projector_state = checkpoint.get("prefix_projector_state")
    if prefix_projectors is not None and isinstance(prefix_projector_state, dict):
        for key, state in prefix_projector_state.items():
            if key in prefix_projectors:
                prefix_projectors[key].load_state_dict(state)
    elif prefix_projector is not None and prefix_projector_state is not None:
        prefix_projector.load_state_dict(prefix_projector_state)
    elif (prefix_projector is not None or prefix_projectors is not None) and prefix_projector_state is None:
        print("Warning: checkpoint has no prefix_projector_state, using random init")

    if trainable_base_model:
        base_model_state = checkpoint.get("base_model_state")
        if base_model_module is None or base_model_state is None:
            raise ValueError(
                "Full-finetune resume requires both base_model_module and base_model_state"
            )
        base_model_module.load_state_dict(base_model_state)

    if batches_per_epoch is None:
        if "epoch" not in checkpoint or "next_batch_idx" not in checkpoint:
            raise ValueError(
                "batches_per_epoch is required when checkpoint metadata does not "
                "contain epoch and next_batch_idx"
            )
        schedule = {
            "global_step": checkpoint_step,
            "start_epoch": max(int(checkpoint["epoch"]), 0),
            "skip_batches_in_epoch": max(int(checkpoint["next_batch_idx"]), 0),
        }
    else:
        schedule = compute_resume_schedule(
            checkpoint=checkpoint,
            batches_per_epoch=batches_per_epoch,
            grad_accum_steps=grad_accum_steps,
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        **schedule,
    }


def maybe_build_probe_wandb_table(run, samples: list[dict]):
    if run is None:
        return None
    try:
        import wandb
    except ImportError:
        return None

    table = wandb.Table(columns=["question_id", "gold", "prediction", "correct"])
    for sample in samples:
        table.add_data(
            str(sample["question_id"]),
            sample["gold"],
            sample["prediction"],
            sample["correct"],
        )
    return table


def compute_per_edge_adjacency_stats(
    adjacency,
    graph_loss_fn,
    agent_roles: list[str],
    grad_accum_steps: int = 1,
) -> dict:
    """Capture per-edge adjacency dynamics at each optimizer step.

    Returns a dict with:
      - edges: list of {src, dst, weight, total_grad, graph_grad, task_grad, sigmoid_deriv}
      - summary: aggregate stats
    """
    with torch.no_grad():
        A = adjacency.get_adjacency()
        logits = adjacency.logits
        total_grad = logits.grad.clone() if logits.grad is not None else torch.zeros_like(logits)
        allowed = adjacency.allowed_edges_mask

    # Compute graph-loss-only gradient analytically (cheap, no model forward needed)
    logits_detached = logits.detach().clone().requires_grad_(True)
    A_detached = torch.sigmoid(logits_detached)
    graph_loss_only = graph_loss_fn(
        A_detached, adjacency.prior, valid_mask=adjacency.allowed_edges_mask,
    )
    graph_loss_only["loss"].backward()
    graph_grad = logits_detached.grad / grad_accum_steps  # match scaling

    n = A.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if not allowed[i, j]:
                continue
            w = float(A[i, j].item())
            tg = float(total_grad[i, j].item())
            gg = float(graph_grad[i, j].item())
            sig_d = w * (1.0 - w)  # sigmoid derivative
            edges.append({
                "src": agent_roles[i] if i < len(agent_roles) else str(i),
                "dst": agent_roles[j] if j < len(agent_roles) else str(j),
                "weight": round(w, 6),
                "logit": round(float(logits[i, j].item()), 4),
                "total_grad": round(tg, 8),
                "graph_grad": round(gg, 8),
                "task_grad": round(tg - gg, 8),
                "sigmoid_deriv": round(sig_d, 8),
            })

    return {"edges": edges}


def train(config_path: str, max_samples: int | None = None):
    config = load_config(config_path)
    training_cfg = config["training"]
    evaluation_cfg = config.get("evaluation", {})
    probe_cfg = config.get("training_probe", {})
    training_input_mode = training_cfg.get("input_mode", "legacy_plain_with_prefix")
    training_seed = int(training_cfg.get("seed", 42))
    resume_checkpoint = resolve_resume_checkpoint(training_cfg)

    # ── Setup device ──
    device, is_ddp = setup_distributed()
    world_size = dist.get_world_size() if is_ddp else 1
    rank = dist.get_rank() if is_ddp else 0
    set_global_seed(training_seed)

    # ── Build system (each process loads its own copy) ──
    if is_main_process():
        print("Building multi-agent system...")
    system = MultiAgentSystem(config)
    system.to(device)

    if is_main_process():
        total_params = sum(p.numel() for p in system.parameters())
        trainable_params = sum(p.numel() for p in system.get_trainable_params())
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"\nInitial graph:\n{system.log_adjacency()}")
        mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"GPU memory after model load: {mem:.2f} GB")

    # ── Wrap trainable modules with DDP ──
    # We can't wrap the entire system (it has frozen model + complex structure)
    # Instead, wrap only the trainable modules
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        ddp_kwargs = build_ddp_kwargs(device.index)

        if training_cfg.get("train_strategy") == "full_finetune":
            system.base_model.model = DDP(
                system.base_model.model,
                **ddp_kwargs,
            )
        # Wrap compressor(s) with DDP
        if system.compressors is not None:
            for idx in range(len(system.compressors)):
                system.compressors[idx] = DDP(
                    system.compressors[idx],
                    **ddp_kwargs,
                )
        elif system.compressor is not None:
            system.compressor = DDP(
                system.compressor,
                **ddp_kwargs,
            )
        # Wrap prefix_projector(s) with DDP
        if system.prefix_projectors is not None:
            for key in list(system.prefix_projectors.keys()):
                system.prefix_projectors[key] = DDP(
                    system.prefix_projectors[key],
                    **ddp_kwargs,
                )
        elif system.prefix_projector is not None:
            system.prefix_projector = DDP(
                system.prefix_projector,
                **ddp_kwargs,
            )
        # Wrap hidden_projections with DDP
        if system.hidden_projections:
            for key in list(system.hidden_projections.keys()):
                system.hidden_projections[key] = DDP(
                    system.hidden_projections[key],
                    **ddp_kwargs,
                )
        # Adjacency is tiny, sync its gradients manually
        # (DDP overhead not worth it for 25 parameters)

    # ── Dataset ──
    if is_main_process():
        print(f"\nLoading dataset: {training_cfg['task']}...")
    loader_kwargs = {}
    if training_cfg.get("source"):
        loader_kwargs["source"] = training_cfg["source"]
    dataset = create_dataset(
        task=training_cfg["task"],
        split="train",
        max_samples=max_samples,
        **loader_kwargs,
    )

    # ── Filter out samples exceeding max_seq_len ──
    max_seq_len = training_cfg.get("max_seq_len", 2048)
    tokenizer = system.base_model.tokenizer
    valid_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        q_len = len(tokenizer.encode(sample["question"], add_special_tokens=True))
        a_len = len(tokenizer.encode(sample["answer"], add_special_tokens=False))
        if q_len + a_len <= max_seq_len:
            valid_indices.append(i)
    n_filtered = len(dataset) - len(valid_indices)
    if n_filtered > 0:
        if is_main_process():
            print(f"Filtered {n_filtered} samples exceeding max_seq_len={max_seq_len} "
                  f"({len(valid_indices)}/{len(dataset)} remaining)")
        dataset = Subset(dataset, valid_indices)

    probe_dataset = None
    probe_indices = []
    if probe_cfg.get("enabled", False):
        dataset, probe_dataset, probe_indices = split_training_and_probe_subsets(
            dataset,
            probe_samples=int(probe_cfg.get("samples", 100)),
            seed=int(probe_cfg.get("seed", 42)),
        )
    drop_last = training_cfg.get("drop_last", True)
    train_shuffle = bool(training_cfg.get("shuffle", True))
    validate_min_samples_for_batches(
        dataset_size=len(dataset),
        per_gpu_batch_size=training_cfg["batch_size"],
        world_size=world_size,
        drop_last=drop_last,
    )

    # DistributedSampler splits data across GPUs
    sampler = (
        DistributedSampler(dataset, shuffle=train_shuffle, seed=training_seed)
        if is_ddp
        else None
    )
    dataloader_generator = None
    if sampler is None and train_shuffle:
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(training_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],  # per-GPU batch size
        shuffle=train_shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=drop_last,
        generator=dataloader_generator,
    )

    if is_main_process():
        print(f"Dataset size: {len(dataset)}")
        if probe_dataset is not None:
            print(f"Probe samples held out: {len(probe_dataset)}")
        print(f"Training seed: {training_seed}")
        print(f"Per-GPU batch size: {training_cfg['batch_size']}")
        print(f"Effective batch size: {training_cfg['batch_size'] * world_size}")
        print(f"Batches per epoch per GPU: {len(dataloader)}")

    # ── Optimizer ──
    base_lr = float(training_cfg["lr"])
    adj_lr = float(training_cfg.get("adjacency_lr", base_lr))
    mlp_lr = float(training_cfg.get("mlp_lr", base_lr))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))

    # Base model params (full_finetune only)
    base_params = []
    if training_cfg.get("train_strategy") == "full_finetune":
        base_params.extend(system.base_model.model.parameters())

    # MLP params: compressor + prefix_projector + hidden_projections
    mlp_params = []
    if system.compressors is not None:
        mlp_params.extend(system.compressors.parameters())
    elif system.compressor is not None:
        mlp_params.extend(system.compressor.parameters())
    if system.learnable_prefix_embeddings is not None:
        mlp_params.extend(system.learnable_prefix_embeddings.parameters())
    if system.prefix_projectors is not None:
        mlp_params.extend(system.prefix_projectors.parameters())
    elif system.prefix_projector is not None:
        mlp_params.extend(system.prefix_projector.parameters())
    if system.hidden_projections:
        mlp_params.extend(system.hidden_projections.parameters())

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr, "name": "base_model"})
    if mlp_params:
        param_groups.append({"params": mlp_params, "lr": mlp_lr, "name": "mlp"})

    freeze_topology = bool(config.get("graph", {}).get("freeze_topology", False))
    if not freeze_topology:
        param_groups.append({"params": list(system.adjacency.parameters()), "lr": adj_lr, "name": "adjacency"})
    elif is_main_process():
        print("Adjacency topology frozen — not included in optimizer")
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    if is_main_process():
        if mlp_lr != base_lr:
            print(f"Using separate MLP lr: {mlp_lr} (base lr: {base_lr})")
        if adj_lr != base_lr:
            print(f"Using separate adjacency lr: {adj_lr} (base lr: {base_lr})")

    # ── LR Scheduler ──
    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    total_training_steps = training_cfg["epochs"] * len(dataloader) // max(training_cfg.get("gradient_accumulation_steps", 1), 1)
    scheduler = None
    if warmup_steps > 0:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_training_steps - warmup_steps, 1),
            eta_min=float(base_lr) * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        if is_main_process():
            print(f"LR schedule: {warmup_steps} warmup steps + cosine decay over {total_training_steps} total steps")

    # ── Training loop ──
    grad_accum_steps = training_cfg.get("gradient_accumulation_steps", 1)
    log_interval = training_cfg.get("log_interval", 1)
    save_interval = training_cfg.get("save_interval", 500)
    if system.compressors is not None:
        compressor_modules = [
            (c.module if is_ddp else c) for c in system.compressors
        ]
        compressor = None  # not used for single compressor
    else:
        compressor = (system.compressor.module if is_ddp else system.compressor) if system.compressor is not None else None
        compressor_modules = None
    # Unwrap learnable_prefix_embeddings (not DDP-wrapped; gradients synced manually)
    learnable_prefix_module = system.learnable_prefix_embeddings  # nn.ParameterList or None
    # Unwrap prefix_projector(s) for checkpoint save/load
    if system.prefix_projectors is not None:
        prefix_projector = None
        prefix_projectors_unwrapped = {
            k: (v.module if is_ddp else v)
            for k, v in system.prefix_projectors.items()
        }
    else:
        prefix_projector = system.prefix_projector.module if is_ddp and system.prefix_projector is not None else system.prefix_projector
        prefix_projectors_unwrapped = None
    trainable_base_model = training_cfg.get("train_strategy") == "full_finetune"
    base_model_module = (
        system.base_model.model.module
        if is_ddp and trainable_base_model
        else system.base_model.model
    )
    adjacency = system.adjacency

    resume_state = None
    if resume_checkpoint is not None:
        resume_state = load_training_checkpoint(
            checkpoint_path=resume_checkpoint,
            compressor=compressor,
            adjacency=adjacency,
            optimizer=optimizer,
            prefix_projector=prefix_projector,
            prefix_projectors=prefix_projectors_unwrapped,
            batches_per_epoch=len(dataloader),
            grad_accum_steps=grad_accum_steps,
            base_model_module=base_model_module if trainable_base_model else None,
            trainable_base_model=trainable_base_model,
            learnable_prefix_module=learnable_prefix_module,
        )

    # ── Output directory with timestamp or resume target ──
    output_dir = resolve_training_output_dir(config)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config for reproducibility
        import yaml
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        run_provenance = build_run_provenance(
            config_path=config_path,
            output_dir=output_dir,
            training_seed=training_seed,
            world_size=world_size,
            rank=rank,
            is_ddp=is_ddp,
        )
        write_probe_history_json(output_dir / "run_provenance.json", run_provenance)
        print(f"  Output dir: {output_dir}")
        if resume_state is not None:
            print(
                "  Resume checkpoint: "
                f"{resume_state['checkpoint_path']} "
                f"(step={resume_state['global_step']}, "
                f"epoch={resume_state['start_epoch']}, "
                f"skip_batches={resume_state['skip_batches_in_epoch']})"
            )

    wandb_run = init_wandb_run(config=config, output_dir=output_dir, rank=rank)

    # ── Loss log ──
    loss_log = []  # list of dicts, saved to CSV
    adjacency_log = []  # per-edge dynamics
    probe_history = []
    probe_history_path = output_dir / "probe_history.json"
    probe_metadata_path = output_dir / "probe_split.json"
    if is_main_process() and probe_dataset is not None:
        write_probe_history_json(
            probe_metadata_path,
            [
                {
                    "probe_indices": probe_indices,
                    "probe_size": len(probe_dataset),
                    "seed": int(probe_cfg.get("seed", 42)),
                }
            ],
        )

    global_step = resume_state["global_step"] if resume_state is not None else 0
    start_epoch = resume_state["start_epoch"] if resume_state is not None else 0
    skip_batches_in_start_epoch = (
        resume_state["skip_batches_in_epoch"] if resume_state is not None else 0
    )
    # ── Mixed Precision ──
    use_amp = bool(training_cfg.get("use_amp", True))
    amp_dtype = torch.bfloat16 if use_amp else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if is_main_process() and use_amp:
        print(f"Mixed precision enabled (dtype={amp_dtype})")

    train_start = time.time()
    total_batches = training_cfg["epochs"] * len(dataloader)
    for epoch in range(start_epoch, training_cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)  # ensure different shuffling each epoch

        system.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if epoch == start_epoch and batch_idx < skip_batches_in_start_epoch:
                continue

            t0 = time.time()
            comp_grad = None
            adj_grad = None
            proj_grad = None
            adj_stats = None
            mem = None

            # ── Tokenize ──
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=training_cfg.get("max_seq_len", 2048),
            )
            task_token_ids = tokenized["input_ids"].to(device)
            task_attention_mask = tokenized["attention_mask"].to(device)

            answer_tokenized = system.base_model.tokenize(
                batch.get("raw_answers", batch["answers"]),
                max_length=2048,
                add_special_tokens=training_input_mode != "chat_with_prefix",
            )
            answer_ids = answer_tokenized["input_ids"].to(device)
            answer_mask = answer_tokenized["attention_mask"].to(device)
            if training_input_mode == "chat_with_prefix":
                answer_ids, answer_mask = append_eos_token(
                    input_ids=answer_ids,
                    attention_mask=answer_mask,
                    eos_token_id=system.base_model.tokenizer.eos_token_id,
                )

            t1 = time.time()

            # ── Forward ──
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                output = system(
                    task_token_ids=task_token_ids,
                    task_attention_mask=task_attention_mask,
                    answer_ids=answer_ids,
                    answer_mask=answer_mask,
                )

            t2 = time.time()

            # ── Backward ──
            loss = output["loss"] / grad_accum_steps
            scaler.scale(loss).backward()

            t3 = time.time()

            # Release references to intermediate tensors (logits, hidden states)
            # to allow CUDA to reclaim memory before the next micro-batch.
            loss_val = output["loss"].item()
            task_loss_val = output["task_loss"].item()
            graph_loss_val = output["graph_loss"].item()
            graph_loss_bce_val = output["graph_loss_bce"].item()
            graph_loss_sparse_val = output["graph_loss_sparse"].item()
            graph_loss_conc_val = output["graph_loss_concentrate"].item()
            del loss, output
            torch.cuda.empty_cache()

            if (batch_idx + 1) % grad_accum_steps == 0:
                # ── Sync adjacency gradients across GPUs (manual allreduce) ──
                if is_ddp and not freeze_topology and adjacency.logits.grad is not None:
                    dist.all_reduce(adjacency.logits.grad, op=dist.ReduceOp.AVG)
                # ── Sync learnable_prefix_embeddings gradients across GPUs ──
                if is_ddp and learnable_prefix_module is not None:
                    for emb in learnable_prefix_module:
                        if emb.grad is not None:
                            dist.all_reduce(emb.grad, op=dist.ReduceOp.AVG)
                mem = torch.cuda.max_memory_allocated(device) / 1024**3
                scaler.unscale_(optimizer)
                # Compute grad norms AFTER unscale so they reflect true magnitudes
                if compressor_modules is not None:
                    comp_grad = compute_grad_norm([p for c in compressor_modules for p in c.parameters()])
                elif compressor is not None:
                    comp_grad = compute_grad_norm(compressor.parameters())
                elif learnable_prefix_module is not None:
                    comp_grad = compute_grad_norm(learnable_prefix_module.parameters())
                else:
                    comp_grad = 0.0
                adj_grad = compute_grad_norm([adjacency.logits])
                if prefix_projectors_unwrapped is not None:
                    proj_grad = compute_grad_norm(
                        [p for pp in prefix_projectors_unwrapped.values() for p in pp.parameters()]
                    )
                elif prefix_projector is not None:
                    proj_grad = compute_grad_norm(prefix_projector.parameters())
                else:
                    proj_grad = 0.0
                # ── Per-edge adjacency monitoring ──
                adj_stats = None
                if is_main_process():
                    adj_stats = compute_per_edge_adjacency_stats(
                        adjacency=adjacency,
                        graph_loss_fn=system.graph_loss_fn,
                        agent_roles=system.agent_roles,
                        grad_accum_steps=grad_accum_steps,
                    )

                torch.nn.utils.clip_grad_norm_(system.get_trainable_params(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if is_main_process():
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_wandb(
                        wandb_run,
                        {
                            "train/global_step": global_step,
                            "train/loss": loss_val,
                            "train/task_loss": task_loss_val,
                            "train/graph_loss": graph_loss_val,
                            "train/graph_loss_bce": graph_loss_bce_val,
                            "train/graph_loss_sparse": graph_loss_sparse_val,
                            "train/graph_loss_concentrate": graph_loss_conc_val,
                            "train/comp_grad": comp_grad,
                            "train/proj_grad": proj_grad,
                            "train/adj_grad": adj_grad,
                            "train/forward_seconds": t2 - t1,
                            "train/backward_seconds": t3 - t2,
                            "train/tokenize_seconds": t1 - t0,
                            "train/max_memory_gb": mem,
                            "train/epoch": epoch + 1,
                            "train/batch": batch_idx + 1,
                            "train/lr": current_lr,
                        },
                        step=global_step,
                    )
                probe_result = None
                if should_run_training_probe(global_step, probe_cfg):
                    probe_result = evaluate_training_probe(
                        system=system,
                        dataset=probe_dataset,
                        config=config,
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        batch_size=int(probe_cfg.get("batch_size", 1)),
                        max_new_tokens=int(probe_cfg.get("max_new_tokens", 64)),
                        inference_mode=evaluation_cfg.get("inference_mode", "chat_with_prefix"),
                        use_terminal_prefix=evaluation_cfg.get("use_terminal_prefix", True),
                        write_agent_logs=bool(probe_cfg.get("write_agent_logs", False)),
                        degeneracy_ratio=float(probe_cfg.get("degenerate_max_new_tokens_ratio", 0.5)),
                    )
                if is_main_process() and probe_result is not None:
                    probe_record = {
                        "global_step": global_step,
                        "metrics": copy.deepcopy(probe_result["metrics"]),
                    }
                    if probe_cfg.get("write_predictions_json", False):
                        probe_record["samples"] = copy.deepcopy(probe_result["samples"])
                    probe_history.append(probe_record)
                    write_probe_history_json(probe_history_path, probe_history)
                    print(
                        f"  Probe@{global_step} | "
                        f"Acc:{probe_result['metrics']['accuracy']:.2f}% "
                        f"({probe_result['metrics']['correct']}/{probe_result['metrics']['total']}) | "
                        f"max-token:{probe_result['metrics']['max_new_tokens_ratio']:.2%} | "
                        f"time:{probe_result['metrics']['time_seconds']:.1f}s"
                    )
                    if probe_result["metrics"]["degenerate"]:
                        print(
                            "  Probe warning: generation is hitting max_new_tokens too often; "
                            "this run may be degenerate."
                        )
                    probe_payload = {
                        "probe/accuracy": probe_result["metrics"]["accuracy"],
                        "probe/correct": probe_result["metrics"]["correct"],
                        "probe/total": probe_result["metrics"]["total"],
                        "probe/time_seconds": probe_result["metrics"]["time_seconds"],
                        "probe/avg_sample_seconds": probe_result["metrics"]["avg_sample_seconds"],
                        "probe/max_new_tokens_count": probe_result["metrics"]["max_new_tokens_count"],
                        "probe/max_new_tokens_ratio": probe_result["metrics"]["max_new_tokens_ratio"],
                        "probe/degenerate": int(probe_result["metrics"]["degenerate"]),
                    }
                    table = maybe_build_probe_wandb_table(wandb_run, probe_result["samples"])
                    if table is not None:
                        probe_payload["probe/samples"] = table
                    log_wandb(
                        wandb_run,
                        probe_payload,
                        step=global_step,
                    )
                optimizer.zero_grad()

            epoch_loss += loss_val
            epoch_batches += 1

            # ── Logging (main process only) ──
            if is_main_process() and (batch_idx + 1) % log_interval == 0:
                display_comp_grad = comp_grad if comp_grad is not None else (
                    compute_grad_norm([p for c in compressor_modules for p in c.parameters()])
                    if compressor_modules is not None
                    else compute_grad_norm(compressor.parameters())
                    if compressor is not None
                    else compute_grad_norm(learnable_prefix_module.parameters())
                    if learnable_prefix_module is not None
                    else 0.0
                )
                if proj_grad is not None:
                    display_proj_grad = proj_grad
                elif prefix_projectors_unwrapped is not None:
                    display_proj_grad = compute_grad_norm(
                        [p for pp in prefix_projectors_unwrapped.values() for p in pp.parameters()]
                    )
                elif prefix_projector is not None:
                    display_proj_grad = compute_grad_norm(prefix_projector.parameters())
                else:
                    display_proj_grad = 0.0
                display_adj_grad = adj_grad if adj_grad is not None else compute_grad_norm([adjacency.logits])
                display_mem = mem if mem is not None else torch.cuda.max_memory_allocated(device) / 1024**3
                completed_batches = epoch * len(dataloader) + (batch_idx + 1)
                elapsed_seconds = time.time() - train_start
                avg_batch_seconds = elapsed_seconds / max(completed_batches, 1)
                eta_seconds = avg_batch_seconds * max(total_batches - completed_batches, 0)

                print(
                    f"  E{epoch+1} B{batch_idx+1}/{len(dataloader)} | "
                    f"Loss:{loss_val:.4f} "
                    f"Task:{task_loss_val:.4f} "
                    f"Graph:{graph_loss_val:.4f} | "
                    f"C∇:{display_comp_grad:.6f} P∇:{display_proj_grad:.6f} A∇:{display_adj_grad:.6f} | "
                    f"tok:{t1-t0:.1f}s fwd:{t2-t1:.1f}s bwd:{t3-t2:.1f}s | "
                    f"mem:{display_mem:.1f}GB | "
                    f"elapsed:{format_duration(elapsed_seconds)} "
                    f"eta:{format_duration(eta_seconds)}"
                )
                # Print per-edge adjacency dynamics
                if adj_stats is not None:
                    edge_strs = []
                    for e in adj_stats["edges"]:
                        edge_strs.append(
                            f"    {e['src']:>8s}→{e['dst']:<8s} "
                            f"w={e['weight']:.4f} logit={e['logit']:+.2f} "
                            f"∇total={e['total_grad']:+.2e} "
                            f"∇task={e['task_grad']:+.2e} "
                            f"∇graph={e['graph_grad']:+.2e} "
                            f"σ'={e['sigmoid_deriv']:.2e}"
                        )
                    print("  Adjacency edges:")
                    print("\n".join(edge_strs))
                
            # ── Record loss ──
            if is_main_process():
                loss_log.append({
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "global_step": global_step,
                    "loss": loss_val,
                    "task_loss": task_loss_val,
                    "graph_loss": graph_loss_val,
                    "comp_grad": comp_grad,
                    "proj_grad": proj_grad,
                    "adj_grad": adj_grad,
                })
                # Record per-edge adjacency dynamics
                if adj_stats is not None:
                    adjacency_log.append({
                        "global_step": global_step,
                        "task_loss": task_loss_val,
                        "graph_loss": graph_loss_val,
                        "edges": adj_stats["edges"],
                    })

            # ── Checkpoint (main process only) ──
            if is_main_process() and should_save_checkpoint(global_step=global_step, save_interval=save_interval):
                ckpt_path = output_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "next_batch_idx": batch_idx + 1,
                    "base_model_state": base_model_module.state_dict() if trainable_base_model else None,
                    "compressor_state": (
                        [c.state_dict() for c in compressor_modules]
                        if compressor_modules is not None
                        else compressor.state_dict() if compressor is not None
                        else None
                    ),
                    "learnable_prefix_state": (
                        learnable_prefix_module.state_dict()
                        if learnable_prefix_module is not None else None
                    ),                    "prefix_projector_state": (
                        {k: v.state_dict() for k, v in prefix_projectors_unwrapped.items()}
                        if prefix_projectors_unwrapped is not None
                        else prefix_projector.state_dict() if prefix_projector is not None
                        else None
                    ),
                    "adjacency_state": adjacency.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}")

        # ── End of epoch ──
        if is_main_process():
            avg_loss = epoch_loss / max(epoch_batches, 1)
            elapsed_seconds = time.time() - train_start
            remaining_epochs = training_cfg["epochs"] - (epoch + 1)
            avg_epoch_seconds = elapsed_seconds / max(epoch + 1, 1)
            eta_seconds = avg_epoch_seconds * remaining_epochs
            print(f"\n{'='*60}")
            print(
                f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | "
                f"elapsed: {format_duration(elapsed_seconds)} | "
                f"eta: {format_duration(eta_seconds)}"
            )
            print(system.log_adjacency())
            A = adjacency.get_adjacency().detach()
            print(f"Adjacency range: [{A.min().item():.4f}, {A.max().item():.4f}]")
            print(f"{'='*60}\n")
            log_wandb(
                wandb_run,
                {
                    "epoch/avg_loss": avg_loss,
                    "epoch/adjacency_min": A.min().item(),
                    "epoch/adjacency_max": A.max().item(),
                    "epoch/index": epoch + 1,
                },
                step=global_step,
            )
            
            csv_path = output_dir / "loss_log.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=loss_log[0].keys())
                writer.writeheader()
                writer.writerows(loss_log)

            # Save per-edge adjacency dynamics
            adj_log_path = output_dir / "adjacency_log.json"
            with open(adj_log_path, "w", encoding="utf-8") as f:
                json.dump(adjacency_log, f, indent=2, ensure_ascii=False)

    eval_summaries = {}
    if evaluation_cfg.get("run_after_train", False):
        eval_kwargs = {
            "config_path": config_path,
            "output_dir": output_dir,
            "checkpoint_path": None,
            "max_new_tokens": evaluation_cfg.get("max_new_tokens", 64),
            "inference_mode": evaluation_cfg.get("inference_mode", "chat_with_prefix"),
            "use_terminal_prefix": evaluation_cfg.get("use_terminal_prefix", True),
            "run_baseline": False,
            "do_sample": evaluation_cfg.get("do_sample", False),
            "write_agent_logs": evaluation_cfg.get("write_agent_logs", False),
            "worker": None,
            "batch_size": evaluation_cfg.get("batch_size", 1),
            "device": device,
            "rank": rank,
            "world_size": world_size,
            "is_dist": is_ddp,
            "cleanup_distributed": False,
        }
        for split_name, sample_limit in resolve_post_train_eval_plan(evaluation_cfg):
            if is_main_process():
                print(f"\nStarting post-train evaluation on {split_name} split...")
            eval_summary = evaluate_loaded_system(
                system=system,
                config=config,
                max_samples=sample_limit,
                split=split_name,
                **eval_kwargs,
            )
            if is_main_process() and eval_summary is not None:
                eval_summaries[split_name] = eval_summary

    # ── Final save ──
    if is_main_process():
        if training_cfg.get("save_final_checkpoint", True):
            final_path = output_dir / "final_model.pt"
            torch.save({
                "step": global_step,
                "epoch": training_cfg["epochs"],
                "next_batch_idx": 0,
                "base_model_state": base_model_module.state_dict() if trainable_base_model else None,
                "compressor_state": (
                    [c.state_dict() for c in compressor_modules]
                    if compressor_modules is not None
                    else compressor.state_dict() if compressor is not None
                    else None
                ),
                "learnable_prefix_state": (
                    learnable_prefix_module.state_dict()
                    if learnable_prefix_module is not None else None
                ),
                "prefix_projector_state": (
                    {k: v.state_dict() for k, v in prefix_projectors_unwrapped.items()}
                    if prefix_projectors_unwrapped is not None
                    else prefix_projector.state_dict() if prefix_projector is not None
                    else None
                ),
                "adjacency_state": adjacency.state_dict(),
                "config": config,
            }, final_path)
            print(f"Training complete. Final model saved to {final_path}")
        else:
            print("Training complete. Final checkpoint save disabled by config.")
        log_wandb(
            wandb_run,
            {
                "final/global_step": global_step,
                "final/output_dir": str(output_dir),
                "final/probe_accuracy": (
                    probe_history[-1]["metrics"]["accuracy"] if probe_history else None
                ),
                "final/train_accuracy": (
                    eval_summaries.get("train", {})
                    .get("metrics", {})
                    .get("accuracy")
                ),
                "final/test_accuracy": (
                    eval_summaries.get("test", {})
                    .get("metrics", {})
                    .get("accuracy")
                ),
            },
            step=global_step,
        )
    finish_wandb(wandb_run)

    cleanup_distributed()
    return {
        "output_dir": str(output_dir),
        "eval_summaries": eval_summaries if is_main_process() else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.max_samples)
