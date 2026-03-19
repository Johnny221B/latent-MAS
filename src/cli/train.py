"""
Training entry point for the multi-agent latent communication system.

Usage:
    python src/cli/train.py --config configs/experiments/gsm8k_3agent.yaml
    python src/cli/train.py --config configs/experiments/gsm8k_3agent.yaml --max_samples 100
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.utils.config import load_config
from src.utils.reporting import finish_wandb, init_wandb_run, log_wandb
from src.utils.training import validate_min_samples_for_batches
from src.pipeline.multi_agent_system import MultiAgentSystem
from data.dataset import create_dataset


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def train(config_path: str, max_samples: int | None = None):
    config = load_config(config_path)
    training_cfg = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build system ──
    print("Building multi-agent system...")
    system = MultiAgentSystem(config)
    system.to(device)

    total_params = sum(p.numel() for p in system.parameters())
    trainable_params = sum(p.numel() for p in system.get_trainable_params())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"\nInitial graph:\n{system.log_adjacency()}")

    # ── Dataset ──
    print(f"\nLoading dataset: {training_cfg['task']}...")
    dataset = create_dataset(
        task=training_cfg["task"],
        split="train",
        max_samples=max_samples,
    )
    validate_min_samples_for_batches(
        dataset_size=len(dataset),
        per_gpu_batch_size=training_cfg["batch_size"],
        world_size=1,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")

    # ── Optimizer ──
    optimizer = AdamW(
        system.get_trainable_params(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )

    # ── Training loop ──
    grad_accum_steps = training_cfg.get("gradient_accumulation_steps", 1)
    log_interval = training_cfg.get("log_interval", 50)
    save_interval = training_cfg.get("save_interval", 500)
    output_dir = Path(config.get("output", {}).get("dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb_run(config=config, output_dir=output_dir, rank=0)

    global_step = 0
    for epoch in range(training_cfg["epochs"]):
        system.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # ── Tokenize question ──
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=training_cfg.get("max_seq_len", 512),
            )
            task_token_ids = tokenized["input_ids"].to(device)
            task_attention_mask = tokenized["attention_mask"].to(device)

            # ── Tokenize answer ──
            answer_tokenized = system.base_model.tokenize(
                batch["answers"],
                max_length=2048,
            )
            answer_ids = answer_tokenized["input_ids"].to(device)
            answer_mask = answer_tokenized["attention_mask"].to(device)

            # ── Forward (teacher forcing) ──
            # Non-terminal agents: latent reasoning → compress → prefix
            # Terminal agent sees: [prefix; role; question; answer]
            # Loss computed only on answer positions
            output = system(
                task_token_ids=task_token_ids,
                task_attention_mask=task_attention_mask,
                answer_ids=answer_ids,
                answer_mask=answer_mask,
            )

            loss = output["loss"] / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(system.get_trainable_params(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += output["loss"].item()

            # ── Logging ──
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)

                # Monitor gradient norms per component
                comp_grad_norm = torch.nn.utils.clip_grad_norm_(
                    system.compressor.parameters(), float('inf')
                )
                adj_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [system.adjacency.logits], float('inf')
                )

                print(
                    f"Epoch {epoch+1}/{training_cfg['epochs']} | "
                    f"Step {global_step} | "
                    f"Batch {batch_idx+1}/{len(dataloader)} | "
                    f"Loss: {output['loss'].item():.4f} (avg: {avg_loss:.4f}) | "
                    f"Task: {output['task_loss'].item():.4f} | "
                    f"Graph: {output['graph_loss'].item():.4f} | "
                    f"Comp∇: {comp_grad_norm:.6f} | "
                    f"Adj∇: {adj_grad_norm:.6f}"
                )
                log_wandb(
                    wandb_run,
                    {
                        "train/loss": output["loss"].item(),
                        "train/task_loss": output["task_loss"].item(),
                        "train/graph_loss": output["graph_loss"].item(),
                        "train/graph_loss_add": output["graph_loss_add"].item(),
                        "train/graph_loss_drop": output["graph_loss_drop"].item(),
                        "train/graph_loss_sparse": output["graph_loss_sparse"].item(),
                        "train/comp_grad": float(comp_grad_norm),
                        "train/adj_grad": float(adj_grad_norm),
                        "train/epoch": epoch + 1,
                        "train/batch": batch_idx + 1,
                    },
                    step=global_step,
                )

            # ── Save checkpoint ──
            if global_step > 0 and global_step % save_interval == 0:
                ckpt_path = output_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "compressor_state": system.compressor.state_dict(),
                    "adjacency_state": system.adjacency.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # ── End of epoch ──
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")
        print(system.log_adjacency())
        print(f"{'='*60}\n")
        A = system.adjacency.get_adjacency().detach()
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

    # ── Final save ──
    final_path = output_dir / "final_model.pt"
    torch.save({
        "step": global_step,
        "compressor_state": system.compressor.state_dict(),
        "adjacency_state": system.adjacency.state_dict(),
        "config": config,
    }, final_path)
    print(f"Training complete. Final model saved to {final_path}")
    log_wandb(
        wandb_run,
        {
            "final/global_step": global_step,
            "final/output_dir": str(output_dir),
        },
        step=global_step,
    )
    finish_wandb(wandb_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-agent latent communication system")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size (for debugging)")
    args = parser.parse_args()

    train(args.config, args.max_samples)
