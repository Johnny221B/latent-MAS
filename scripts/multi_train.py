"""
Training entry point with proper multi-GPU DDP support.

Single GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/experiments/gsm8k_3agent.yaml

Multi-GPU (8 cards):
    torchrun --nproc_per_node=8 scripts/train.py --config configs/experiments/gsm8k_3agent.yaml

Specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train.py --config configs/experiments/gsm8k_3agent.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path
import csv
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW

from src.utils.config import load_config
from src.pipeline.multi_agent_system import MultiAgentSystem
from data.dataset import create_dataset


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


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

        # Wrap compressor with DDP
        system.compressor = DDP(
            system.compressor,
            device_ids=[device.index],
            find_unused_parameters=True,
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

    # DistributedSampler splits data across GPUs
    sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],  # per-GPU batch size
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
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
    output_dir = Path(config.get("output", {}).get("dir", "outputs"))
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    compressor = system.compressor.module if is_ddp else system.compressor
    adjacency = system.adjacency
    
    # ── Output directory with timestamp ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = config.get("output", {}).get("dir", "outputs/run")
    output_dir = Path(f"{base_output_dir}_{timestamp}")
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config for reproducibility
        import yaml
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"  Output dir: {output_dir}")

    # ── Loss log ──
    loss_log = []  # list of dicts, saved to CSV

    global_step = 0
    for epoch in range(training_cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)  # ensure different shuffling each epoch

        system.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            t0 = time.time()

            # ── Tokenize ──
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=training_cfg.get("max_seq_len", 256),
            )
            task_token_ids = tokenized["input_ids"].to(device)
            task_attention_mask = tokenized["attention_mask"].to(device)

            answer_tokenized = system.base_model.tokenize(
                batch["answers"],
                max_length=128,
            )
            answer_ids = answer_tokenized["input_ids"].to(device)
            answer_mask = answer_tokenized["attention_mask"].to(device)

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
                torch.nn.utils.clip_grad_norm_(system.get_trainable_params(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += output["loss"].item()
            epoch_batches += 1

            # ── Logging (main process only) ──
            if is_main_process() and (batch_idx + 1) % log_interval == 0:
                comp_grad = sum(
                    p.grad.norm().item() for p in compressor.parameters()
                    if p.grad is not None
                )
                adj_grad = (
                    adjacency.logits.grad.norm().item()
                    if adjacency.logits.grad is not None else 0.0
                )
                mem = torch.cuda.max_memory_allocated(device) / 1024**3

                print(
                    f"  E{epoch+1} B{batch_idx+1}/{len(dataloader)} | "
                    f"Loss:{output['loss'].item():.4f} "
                    f"Task:{output['task_loss'].item():.4f} "
                    f"Graph:{output['graph_loss'].item():.4f} | "
                    f"C∇:{comp_grad:.6f} A∇:{adj_grad:.6f} | "
                    f"tok:{t1-t0:.1f}s fwd:{t2-t1:.1f}s bwd:{t3-t2:.1f}s | "
                    f"mem:{mem:.1f}GB"
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
                    "comp_grad": comp_grad if (batch_idx + 1) % log_interval == 0 else None,
                    "adj_grad": adj_grad if (batch_idx + 1) % log_interval == 0 else None,
                })

            # ── Checkpoint (main process only) ──
            if is_main_process() and global_step > 0 and global_step % save_interval == 0:
                ckpt_path = output_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "compressor_state": compressor.state_dict(),
                    "adjacency_state": adjacency.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}")

        # ── End of epoch ──
        if is_main_process():
            avg_loss = epoch_loss / max(epoch_batches, 1)
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")
            print(system.log_adjacency())
            A = adjacency.get_adjacency().detach()
            print(f"Adjacency range: [{A.min().item():.4f}, {A.max().item():.4f}]")
            print(f"{'='*60}\n")
            
            csv_path = output_dir / "loss_log.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=loss_log[0].keys())
                writer.writeheader()
                writer.writerows(loss_log)

    # ── Final save ──
    if is_main_process():
        final_path = output_dir / "final_model.pt"
        torch.save({
            "step": global_step,
            "compressor_state": compressor.state_dict(),
            "adjacency_state": adjacency.state_dict(),
            "config": config,
        }, final_path)
        print(f"Training complete. Final model saved to {final_path}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.max_samples)