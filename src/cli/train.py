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
import os
import sys
import time
from pathlib import Path
import csv
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW

from src.utils.config import load_config
from src.utils.output_paths import build_timestamped_output_dir
from src.utils.reporting import finish_wandb, init_wandb_run, log_wandb
from src.utils.token_utils import append_eos_token
from src.utils.training import (
    build_ddp_kwargs,
    compute_grad_norm,
    should_save_checkpoint,
    validate_min_samples_for_batches,
)
from src.pipeline.multi_agent_system import MultiAgentSystem
from src.cli.evaluate import evaluate_loaded_system
from data.dataset import create_dataset


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


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


def train(config_path: str, max_samples: int | None = None):
    config = load_config(config_path)
    training_cfg = config["training"]
    evaluation_cfg = config.get("evaluation", {})
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
        }
        for split_name, sample_limit in (
            ("train", evaluation_cfg.get("train_probe_samples")),
            ("test", evaluation_cfg.get("test_probe_samples")),
        ):
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
