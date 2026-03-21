#!/usr/bin/env python3
"""
Run eval-only text-message DAG communication without retraining.

This is intended for direct comparisons against latent-prefix eval runs when no
checkpoint was saved for a prior training iteration. The routing graph comes
from the current config / initialization, not from a loaded checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.cli.evaluate import cleanup_eval_distributed, evaluate_loaded_system, setup_eval_distributed
from src.pipeline.multi_agent_system import MultiAgentSystem
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/gsm8k_5agent_probe64_text_dag_eval.yaml",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=12000)
    parser.add_argument("--text-message-edge-threshold", type=float, default=0.5)
    parser.add_argument("--text-message-max-new-tokens", type=int, default=512)
    parser.add_argument("--preview-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"qwen3_probe64_text_eval_only_iter05route_{stamp}"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg["run_after_train"] = False
    eval_cfg["communication_mode"] = "text_messages"
    eval_cfg["write_agent_logs"] = True
    eval_cfg["train_probe_samples"] = 0
    eval_cfg["batch_size"] = args.batch_size
    eval_cfg["max_new_tokens"] = args.max_new_tokens
    eval_cfg["text_message_edge_threshold"] = args.text_message_edge_threshold
    eval_cfg["text_message_max_new_tokens"] = args.text_message_max_new_tokens
    if args.split == "test":
        eval_cfg["test_probe_samples"] = args.max_samples

    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "mode": "eval_only_text_messages",
        "comparison_basis": (
            "iter05 did not save a checkpoint; this run uses the same "
            "prior-residual graph configuration for eval-only text-message routing."
        ),
        "config_path": args.config,
        "split": args.split,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "text_message_edge_threshold": args.text_message_edge_threshold,
        "text_message_max_new_tokens": args.text_message_max_new_tokens,
        "preview_limit": args.preview_limit,
    }
    with open(output_dir / "eval_context.json", "w") as f:
        json.dump(context, f, indent=2)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    device, rank, world_size, is_dist = setup_eval_distributed()

    try:
        system = MultiAgentSystem(config)
        result = evaluate_loaded_system(
            system=system,
            config=config,
            config_path=args.config,
            output_dir=output_dir,
            checkpoint_path=None,
            max_samples=args.max_samples,
            split=args.split,
            max_new_tokens=args.max_new_tokens,
            inference_mode="chat_with_prefix",
            use_terminal_prefix=False,
            communication_mode="text_messages",
            text_message_edge_threshold=args.text_message_edge_threshold,
            text_message_max_new_tokens=args.text_message_max_new_tokens,
            run_baseline=False,
            do_sample=False,
            write_agent_logs=True,
            worker=None,
            batch_size=args.batch_size,
            preview_limit=args.preview_limit,
            device=device,
            rank=rank,
            world_size=world_size,
            is_dist=is_dist,
        )
        if rank == 0 and result is not None:
            print("OUTPUT_DIR", output_dir)
            print("RESULT", result["metrics"])
    finally:
        if is_dist:
            cleanup_eval_distributed()


if __name__ == "__main__":
    main()
