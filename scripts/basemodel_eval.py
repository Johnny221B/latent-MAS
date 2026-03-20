"""
Baseline evaluation: test a single LLM (no multi-agent, no prefix) on benchmarks.

This is the simplest baseline — the model directly answers each question.
Use this to compare against the multi-agent system.

Usage:
    # Qwen3-0.6B on GSM8K
    CUDA_VISIBLE_DEVICES=0 python scripts/baseline_eval.py \
        --model_path /data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B \
        --task gsm8k \
        --max_samples 100

    # Qwen3-4B on GSM8K (full test set)
    CUDA_VISIBLE_DEVICES=0 python scripts/baseline_eval.py \
        --model_path /data2/yangyz/latent-MAS/weights/Qwen__Qwen3-4B \
        --task gsm8k

    # Qwen3-0.6B on ARC-Challenge
    CUDA_VISIBLE_DEVICES=0 python scripts/baseline_eval.py \
        --model_path /data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B \
        --task arc_challenge

Supported tasks: gsm8k, arc_easy, arc_challenge, medqa
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.answer_extraction import extract_answer
from data.dataset import create_dataset


# ── Task-specific prompt templates ──
TASK_PROMPTS = {
    "gsm8k": (
        "Solve the following math problem step by step. "
        "Put your final numeric answer after ####.\n\n"
        "Question: {question}\n\n"
        "Solution:"
    ),
    "arc_easy": (
        "Answer the following multiple choice question. "
        "Your final answer must be a single letter (A, B, C, or D).\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "arc_challenge": (
        "Answer the following multiple choice question. "
        "Think step by step, then give your final answer as a single letter (A, B, C, or D).\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "medqa": (
        "Answer the following medical question. "
        "Your final answer must be a single letter (A, B, C, or D).\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
}


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def evaluate(
    model_path: str,
    task: str,
    max_samples: int | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    split: str = "test",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    print(f"  Model: {model.config._name_or_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  dtype: {next(model.parameters()).dtype}")
    mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"  GPU memory: {mem:.2f} GB")

    # ── Load dataset ──
    print(f"\nLoading dataset: {task} ({split})")
    dataset = create_dataset(task=task, split=split, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"  Samples: {len(dataset)}")

    # ── Get prompt template ──
    prompt_template = TASK_PROMPTS.get(task, "Question: {question}\n\nAnswer:")
    print(f"  Prompt template: {prompt_template[:80]}...")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature} ({'greedy' if temperature == 0 else 'sampling'})")

    # ── Evaluate ──
    correct = 0
    total = 0
    results = []

    print(f"\nRunning evaluation...")
    t_start = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            t0 = time.time()

            question = batch["questions"][0]
            gold = batch["answers"][0].strip()

            # Build prompt
            prompt = prompt_template.format(question=question)

            # Tokenize
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Generate
            if temperature == 0:
                gen_out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                gen_out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode only the generated part
            generated_ids = gen_out[0][input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Extract answer
            pred = extract_answer(generated_text, task_type=task).strip()
            is_correct = pred == gold

            if is_correct:
                correct += 1
            total += 1

            t1 = time.time()

            results.append({
                "question": question,
                "gold": gold,
                "prediction": pred,
                "generated_text": generated_text[:500],
                "correct": is_correct,
            })

            # Print examples (first 5)
            if idx < 5:
                status = "✓" if is_correct else "✗"
                print(f"\n  Example {idx+1} [{status}]:")
                print(f"    Q: {question[:100]}...")
                print(f"    Gold: {gold}")
                print(f"    Pred: {pred}")
                print(f"    Gen:  {generated_text[:150]}")

            # Progress
            if (idx + 1) % 50 == 0 or (idx + 1) == len(dataloader):
                acc = correct / total * 100
                elapsed = time.time() - t_start
                eta = elapsed / total * (len(dataloader) - total)
                print(
                    f"  [{idx+1}/{len(dataloader)}] "
                    f"Acc: {acc:.1f}% ({correct}/{total}) | "
                    f"{t1-t0:.1f}s/sample | "
                    f"ETA: {eta/60:.0f}min"
                )

    # ── Results ──
    t_total = time.time() - t_start
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Model:    {model_path}")
    print(f"  Task:     {task} ({split})")
    print(f"  Samples:  {total}")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Time:     {t_total:.0f}s ({t_total/total:.1f}s/sample)")
    print(f"{'='*60}")

    # ── Save results ──
    model_name = Path(model_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/baseline_{model_name}_{task}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_data = {
        "model": model_path,
        "task": task,
        "split": split,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time_seconds": t_total,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "samples": results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    # Save summary (easy to read)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Model:    {model_path}\n")
        f.write(f"Task:     {task} ({split})\n")
        f.write(f"Accuracy: {accuracy:.2f}% ({correct}/{total})\n")
        f.write(f"Time:     {t_total:.0f}s\n")

    print(f"  Results saved: {output_dir}/")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline evaluation of a single LLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to local model weights")
    parser.add_argument("--task", type=str, default="gsm8k",
                        choices=["gsm8k", "arc_easy", "arc_challenge", "medqa"],
                        help="Evaluation task")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit test samples")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="0=greedy, >0=sampling")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        task=args.task,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        split=args.split,
    )