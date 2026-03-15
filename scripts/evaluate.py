"""
Evaluation entry point.

Usage:
    python scripts/evaluate.py --config configs/experiments/gsm8k_3agent.yaml --checkpoint outputs/final_model.pt
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.answer_extraction import extract_answer
from src.pipeline.multi_agent_system import MultiAgentSystem
from data.dataset import create_dataset


def collate_fn(batch):
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def evaluate(config_path: str, checkpoint_path: str, max_samples: int | None = None):
    """Run evaluation."""
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build system
    system = MultiAgentSystem(config)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    system.compressor.load_state_dict(ckpt["compressor_state"])
    system.adjacency.load_state_dict(ckpt["adjacency_state"])
    system.to(device)
    system.eval()

    print(f"Loaded checkpoint from {checkpoint_path} (step {ckpt.get('step', '?')})")
    print(f"\nLearned graph:\n{system.log_adjacency()}")
    print(f"Hard adjacency (threshold=0.5):")
    print(system.adjacency.get_hard_adjacency())

    # Dataset
    task = config["training"]["task"]
    dataset = create_dataset(task=task, split="test", max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=config["training"].get("max_seq_len", 512),
            )
            task_token_ids = tokenized["input_ids"].to(device)
            task_attention_mask = tokenized["attention_mask"].to(device)

            output = system(
                task_token_ids=task_token_ids,
                task_attention_mask=task_attention_mask,
            )

            # Greedy decode from logits
            predicted_ids = output["final_logits"].argmax(dim=-1)  # [B, seq_len]
            predicted_texts = system.base_model.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True,
            )

            for pred_text, gold_answer in zip(predicted_texts, batch["answers"]):
                pred_answer = extract_answer(pred_text, task_type=task)
                if pred_answer.strip() == gold_answer.strip():
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nResults on {task} ({total} samples):")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-agent system")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint, args.max_samples)
