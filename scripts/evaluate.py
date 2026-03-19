"""
Evaluation script: test trained model on GSM8K test set.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
        --config outputs/gsm8k_qwen3-0.6b_20260317_225851/config.yaml \
        --checkpoint outputs/gsm8k_qwen3-0.6b_20260317_225851/final_model.pt \
        --max_samples 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.answer_extraction import extract_answer
from src.pipeline.multi_agent_system import MultiAgentSystem
from data.dataset import create_dataset


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def evaluate(config_path: str, checkpoint_path: str, max_samples: int | None = None):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build system ──
    print("Building multi-agent system...")
    system = MultiAgentSystem(config)

    # ── Load checkpoint ──
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
    print(f"\nLearned adjacency:")
    print(system.log_adjacency())
    A = system.adjacency.get_adjacency().detach()
    print(f"Range: [{A.min().item():.4f}, {A.max().item():.4f}]")
    print(f"Hard adjacency (threshold=0.5):")
    print(system.adjacency.get_hard_adjacency())

    # ── Dataset ──
    task = config["training"]["task"]
    print(f"\nLoading {task} test set...")
    dataset = create_dataset(task=task, split="test", max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Test samples: {len(dataset)}")

    # ── Evaluate ──
    correct = 0
    total = 0
    results = []

    print("\nRunning evaluation...")
    t_start = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            t0 = time.time()

            # Tokenize question
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=config["training"].get("max_seq_len", 256),
            )
            task_ids = tokenized["input_ids"].to(device)
            task_mask = tokenized["attention_mask"].to(device)

            # Run inference (no answer_ids → triggers generate_answer path)
            output = system(
                task_token_ids=task_ids,
                task_attention_mask=task_mask,
            )

            generated_text = output["generated_text"]

            # Extract and compare answer
            pred = extract_answer(generated_text, task_type=task)
            gold = batch["answers"][0].strip()
            is_correct = pred.strip() == gold.strip()

            if is_correct:
                correct += 1
            total += 1

            t1 = time.time()

            results.append({
                "question": batch["questions"][0],
                "gold": gold,
                "prediction": pred,
                "generated_text": generated_text[:500],
                "correct": is_correct,
            })

            # Print progress
            if (idx + 1) % 10 == 0 or (idx + 1) == len(dataloader):
                acc = correct / total * 100
                print(
                    f"  [{idx+1}/{len(dataloader)}] "
                    f"Acc: {acc:.1f}% ({correct}/{total}) | "
                    f"{t1-t0:.1f}s/sample"
                )

            # Print some examples
            if idx < 5:
                status = "✓" if is_correct else "✗"
                print(f"\n  Example {idx+1} [{status}]:")
                print(f"    Q: {batch['questions'][0][:100]}...")
                print(f"    Gold: {gold}")
                print(f"    Pred: {pred}")
                print(f"    Gen:  {generated_text[:200]}")

    # ── Summary ──
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

    # ── Save results ──
    checkpoint_dir = Path(checkpoint_path).parent
    eval_path = checkpoint_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump({
            "task": task,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "time_seconds": t_total,
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {eval_path}")

    # ── Also run baseline (no multi-agent, just the model) ──
    print(f"\n{'='*60}")
    print(f"  BASELINE (single model, no multi-agent)")
    print(f"{'='*60}")

    baseline_correct = 0
    baseline_total = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            tokenized = system.base_model.tokenize(
                batch["questions"],
                max_length=config["training"].get("max_seq_len", 256),
            )
            task_ids = tokenized["input_ids"].to(device)
            task_mask = tokenized["attention_mask"].to(device)

            # Direct generation without any prefix
            gen_out = system.base_model.model.generate(
                input_ids=task_ids,
                attention_mask=task_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=system.base_model.tokenizer.pad_token_id,
            )
            baseline_text = system.base_model.tokenizer.decode(
                gen_out[0][task_ids.shape[1]:], skip_special_tokens=True,
            )

            pred = extract_answer(baseline_text, task_type=task)
            gold = batch["answers"][0].strip()
            if pred.strip() == gold.strip():
                baseline_correct += 1
            baseline_total += 1

            if (idx + 1) % 50 == 0:
                bacc = baseline_correct / baseline_total * 100
                print(f"  [{idx+1}/{len(dataloader)}] Baseline Acc: {bacc:.1f}%")

    baseline_acc = baseline_correct / baseline_total * 100 if baseline_total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Multi-agent (trained): {accuracy:.2f}%")
    print(f"  Single model baseline: {baseline_acc:.2f}%")
    print(f"  Improvement:           {accuracy - baseline_acc:+.2f}%")
    print(f"{'='*60}")

    # Append baseline to results
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
    eval_data["baseline_accuracy"] = baseline_acc
    eval_data["improvement"] = accuracy - baseline_acc
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.max_samples)