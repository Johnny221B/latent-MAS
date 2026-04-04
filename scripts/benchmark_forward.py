"""Benchmark forward pass speed with different optimization strategies.

Usage:
    python scripts/benchmark_forward.py --config configs/experiments/am_deepseek_r1_hier6_4b.yaml --max_samples 100
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Subset

from src.utils.config import load_config
from src.pipeline.multi_agent_system import MultiAgentSystem
from src.utils.token_utils import append_eos_token
from src.data import create_dataset


def collate_fn(batch):
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def benchmark_one_config(
    config: dict,
    dataset,
    device: torch.device,
    max_batches: int = 25,
    label: str = "baseline",
    compile_model: bool = False,
):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}")
    print(f"{'='*60}")

    system = MultiAgentSystem(config)
    system.to(device)

    if compile_model:
        print("Compiling frozen model with torch.compile...")
        system.base_model.model = torch.compile(
            system.base_model.model,
            mode="reduce-overhead",
            dynamic=True,
        )

    training_input_mode = config.get("training", {}).get("input_mode", "legacy_plain_with_prefix")
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate_fn, drop_last=True)

    system.train()

    # Warmup
    print("Warming up (2 batches)...")
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        tokenized = system.base_model.tokenize(batch["questions"], max_length=config["training"].get("max_seq_len", 2048))
        task_ids = tokenized["input_ids"].to(device)
        task_mask = tokenized["attention_mask"].to(device)
        ans_tok = system.base_model.tokenize(batch["answers"], max_length=2048, add_special_tokens=training_input_mode != "chat_with_prefix")
        answer_ids = ans_tok["input_ids"].to(device)
        answer_mask = ans_tok["attention_mask"].to(device)
        if training_input_mode == "chat_with_prefix":
            answer_ids, answer_mask = append_eos_token(answer_ids, answer_mask, system.base_model.tokenizer.eos_token_id)
        with torch.no_grad():
            output = system(task_token_ids=task_ids, task_attention_mask=task_mask, answer_ids=answer_ids, answer_mask=answer_mask)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({max_batches} batches)...")
    fwd_times = []
    total_times = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches + 2:  # skip first 2 warmup
            break
        if i < 2:
            continue

        t_start = time.time()

        tokenized = system.base_model.tokenize(batch["questions"], max_length=config["training"].get("max_seq_len", 2048))
        task_ids = tokenized["input_ids"].to(device)
        task_mask = tokenized["attention_mask"].to(device)
        ans_tok = system.base_model.tokenize(batch["answers"], max_length=2048, add_special_tokens=training_input_mode != "chat_with_prefix")
        answer_ids = ans_tok["input_ids"].to(device)
        answer_mask = ans_tok["attention_mask"].to(device)
        if training_input_mode == "chat_with_prefix":
            answer_ids, answer_mask = append_eos_token(answer_ids, answer_mask, system.base_model.tokenizer.eos_token_id)

        t_fwd_start = time.time()
        output = system(task_token_ids=task_ids, task_attention_mask=task_mask, answer_ids=answer_ids, answer_mask=answer_mask)
        loss = output["loss"]
        loss.backward()
        torch.cuda.synchronize()
        t_end = time.time()

        fwd_times.append(t_end - t_fwd_start)
        total_times.append(t_end - t_start)

    # Cleanup
    del system
    torch.cuda.empty_cache()

    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_total = sum(total_times) / len(total_times)
    mem = torch.cuda.max_memory_allocated(device) / 1024**3

    print(f"\nResults [{label}]:")
    print(f"  Avg fwd+bwd:  {avg_fwd:.3f}s")
    print(f"  Avg total:    {avg_total:.3f}s")
    print(f"  Peak memory:  {mem:.1f} GB")
    print(f"  Batches:      {len(fwd_times)}")

    return {
        "label": label,
        "avg_fwd_bwd": avg_fwd,
        "avg_total": avg_total,
        "peak_memory_gb": mem,
        "fwd_times": fwd_times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = load_config(args.config)

    # Load dataset
    training_cfg = config["training"]
    loader_kwargs = {}
    if training_cfg.get("source"):
        loader_kwargs["source"] = training_cfg["source"]
    dataset = create_dataset(task=training_cfg["task"], split="train", max_samples=args.max_samples, **loader_kwargs)
    print(f"Dataset: {len(dataset)} samples")

    results = []

    # ── Baseline ──
    config_baseline = load_config(args.config)
    r = benchmark_one_config(config_baseline, dataset, device, max_batches=args.max_batches, label="baseline")
    results.append(r)

    # ── D: Reduced steps (40 -> 20) ──
    config_d = load_config(args.config)
    config_d["reasoning"]["steps_per_agent"] = 20
    config_d["reasoning"]["compress_last_k"] = 20
    r = benchmark_one_config(config_d, dataset, device, max_batches=args.max_batches, label="reduced_steps_20")
    results.append(r)

    # ── C: torch.compile ──
    config_c = load_config(args.config)
    r = benchmark_one_config(config_c, dataset, device, max_batches=args.max_batches, label="torch_compile", compile_model=True)
    results.append(r)

    # ── D2: Reduced steps (40 -> 10) ──
    config_d2 = load_config(args.config)
    config_d2["reasoning"]["steps_per_agent"] = 10
    config_d2["reasoning"]["compress_last_k"] = 10
    r = benchmark_one_config(config_d2, dataset, device, max_batches=args.max_batches, label="reduced_steps_10")
    results.append(r)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    baseline_time = results[0]["avg_fwd_bwd"]
    print(f"{'Label':<25} {'Avg fwd+bwd':>12} {'Speedup':>10} {'Memory':>10}")
    print("-" * 60)
    for r in results:
        speedup = baseline_time / r["avg_fwd_bwd"]
        print(f"{r['label']:<25} {r['avg_fwd_bwd']:>10.3f}s {speedup:>9.2f}x {r['peak_memory_gb']:>8.1f}GB")


if __name__ == "__main__":
    main()
