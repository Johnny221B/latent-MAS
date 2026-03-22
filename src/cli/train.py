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
import sys
import time
from pathlib import Path
import csv
from datetime import datetime
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
        },
        "samples": merged_results,
    }


def write_probe_history_json(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


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


def train(config_path: str, max_samples: int | None = None):
    config = load_config(config_path)
    training_cfg = config["training"]
    evaluation_cfg = config.get("evaluation", {})
    probe_cfg = config.get("training_probe", {})
    training_input_mode = training_cfg.get("input_mode", "legacy_plain_with_prefix")

    # ── Setup device ──
    device, is_ddp = setup_distributed()
    world_size = dist.get_world_size() if is_ddp else 1
    rank = dist.get_rank() if is_ddp else 0

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
        # Wrap compressor with DDP
        system.compressor = DDP(
            system.compressor,
            **ddp_kwargs,
        )
        # Adjacency is tiny, sync its gradients manually
        # (DDP overhead not worth it for 25 parameters)

    # ── Dataset ──
    if is_main_process():
        print(f"\nLoading dataset: {training_cfg['task']}...")
    dataset = create_dataset(
        task=training_cfg["task"],
        split="train",
        max_samples=max_samples,
    )
    probe_dataset = None
    probe_indices = []
    if probe_cfg.get("enabled", False):
        dataset, probe_dataset, probe_indices = split_training_and_probe_subsets(
            dataset,
            probe_samples=int(probe_cfg.get("samples", 100)),
            seed=int(probe_cfg.get("seed", 42)),
        )
    drop_last = training_cfg.get("drop_last", True)
    validate_min_samples_for_batches(
        dataset_size=len(dataset),
        per_gpu_batch_size=training_cfg["batch_size"],
        world_size=world_size,
        drop_last=drop_last,
    )

    # DistributedSampler splits data across GPUs
    sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],  # per-GPU batch size
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )

    if is_main_process():
        print(f"Dataset size: {len(dataset)}")
        if probe_dataset is not None:
            print(f"Probe samples held out: {len(probe_dataset)}")
        print(f"Per-GPU batch size: {training_cfg['batch_size']}")
        print(f"Effective batch size: {training_cfg['batch_size'] * world_size}")
        print(f"Batches per epoch per GPU: {len(dataloader)}")

    # ── Optimizer ──
    optimizer = AdamW(
        system.get_trainable_params(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )

    # ── Training loop ──
    grad_accum_steps = training_cfg.get("gradient_accumulation_steps", 1)
    log_interval = training_cfg.get("log_interval", 1)
    save_interval = training_cfg.get("save_interval", 500)
    compressor = system.compressor.module if is_ddp else system.compressor
    trainable_base_model = training_cfg.get("train_strategy") == "full_finetune"
    base_model_module = (
        system.base_model.model.module
        if is_ddp and trainable_base_model
        else system.base_model.model
    )
    adjacency = system.adjacency
    
    # ── Output directory with timestamp ──
    base_output_dir = config.get("output", {}).get("dir", "outputs/run")
    output_dir = build_timestamped_output_dir(base_output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config for reproducibility
        import yaml
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"  Output dir: {output_dir}")

    wandb_run = init_wandb_run(config=config, output_dir=output_dir, rank=rank)

    # ── Loss log ──
    loss_log = []  # list of dicts, saved to CSV
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

    global_step = 0
    train_start = time.time()
    total_batches = training_cfg["epochs"] * len(dataloader)
    for epoch in range(training_cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)  # ensure different shuffling each epoch

        system.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            t0 = time.time()
            comp_grad = None
            adj_grad = None
            mem = None

            # ── Tokenize ──
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=training_cfg.get("max_seq_len", 2048),
            )
            task_token_ids = tokenized["input_ids"].to(device)
            task_attention_mask = tokenized["attention_mask"].to(device)

            answer_tokenized = system.base_model.tokenize(
                batch["answers"],
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
            output = system(
                task_token_ids=task_token_ids,
                task_attention_mask=task_attention_mask,
                answer_ids=answer_ids,
                answer_mask=answer_mask,
            )

            t2 = time.time()

            # ── Backward ──
            loss = output["loss"] / grad_accum_steps
            loss.backward()

            t3 = time.time()

            # ── Sync adjacency gradients across GPUs (manual allreduce) ──
            if is_ddp and adjacency.logits.grad is not None:
                dist.all_reduce(adjacency.logits.grad, op=dist.ReduceOp.AVG)

            if (batch_idx + 1) % grad_accum_steps == 0:
                comp_grad = compute_grad_norm(compressor.parameters())
                adj_grad = compute_grad_norm([adjacency.logits])
                mem = torch.cuda.max_memory_allocated(device) / 1024**3
                torch.nn.utils.clip_grad_norm_(system.get_trainable_params(), 1.0)
                optimizer.step()
                global_step += 1
                if is_main_process():
                    log_wandb(
                        wandb_run,
                        {
                            "train/global_step": global_step,
                            "train/loss": output["loss"].item(),
                            "train/task_loss": output["task_loss"].item(),
                            "train/graph_loss": output["graph_loss"].item(),
                            "train/graph_loss_add": output["graph_loss_add"].item(),
                            "train/graph_loss_drop": output["graph_loss_drop"].item(),
                            "train/graph_loss_sparse": output["graph_loss_sparse"].item(),
                            "train/comp_grad": comp_grad,
                            "train/adj_grad": adj_grad,
                            "train/forward_seconds": t2 - t1,
                            "train/backward_seconds": t3 - t2,
                            "train/tokenize_seconds": t1 - t0,
                            "train/max_memory_gb": mem,
                            "train/epoch": epoch + 1,
                            "train/batch": batch_idx + 1,
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
                        f"time:{probe_result['metrics']['time_seconds']:.1f}s"
                    )
                    probe_payload = {
                        "probe/accuracy": probe_result["metrics"]["accuracy"],
                        "probe/correct": probe_result["metrics"]["correct"],
                        "probe/total": probe_result["metrics"]["total"],
                        "probe/time_seconds": probe_result["metrics"]["time_seconds"],
                        "probe/avg_sample_seconds": probe_result["metrics"]["avg_sample_seconds"],
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

            epoch_loss += output["loss"].item()
            epoch_batches += 1

            # ── Logging (main process only) ──
            if is_main_process() and (batch_idx + 1) % log_interval == 0:
                display_comp_grad = comp_grad if comp_grad is not None else compute_grad_norm(compressor.parameters())
                display_adj_grad = adj_grad if adj_grad is not None else compute_grad_norm([adjacency.logits])
                display_mem = mem if mem is not None else torch.cuda.max_memory_allocated(device) / 1024**3
                completed_batches = epoch * len(dataloader) + (batch_idx + 1)
                elapsed_seconds = time.time() - train_start
                avg_batch_seconds = elapsed_seconds / max(completed_batches, 1)
                eta_seconds = avg_batch_seconds * max(total_batches - completed_batches, 0)

                print(
                    f"  E{epoch+1} B{batch_idx+1}/{len(dataloader)} | "
                    f"Loss:{output['loss'].item():.4f} "
                    f"Task:{output['task_loss'].item():.4f} "
                    f"Graph:{output['graph_loss'].item():.4f} | "
                    f"C∇:{display_comp_grad:.6f} A∇:{display_adj_grad:.6f} | "
                    f"tok:{t1-t0:.1f}s fwd:{t2-t1:.1f}s bwd:{t3-t2:.1f}s | "
                    f"mem:{display_mem:.1f}GB | "
                    f"elapsed:{format_duration(elapsed_seconds)} "
                    f"eta:{format_duration(eta_seconds)}"
                )
                
            # ── Record loss ──
            if is_main_process():
                loss_log.append({
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "global_step": global_step,
                    "loss": output["loss"].item(),
                    "task_loss": output["task_loss"].item(),
                    "graph_loss": output["graph_loss"].item(),
                    "comp_grad": comp_grad,
                    "adj_grad": adj_grad,
                })

            # ── Checkpoint (main process only) ──
            if is_main_process() and should_save_checkpoint(global_step=global_step, save_interval=save_interval):
                ckpt_path = output_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "base_model_state": base_model_module.state_dict() if trainable_base_model else None,
                    "compressor_state": compressor.state_dict(),
                    "adjacency_state": adjacency.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
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
                "base_model_state": base_model_module.state_dict() if trainable_base_model else None,
                "compressor_state": compressor.state_dict(),
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
