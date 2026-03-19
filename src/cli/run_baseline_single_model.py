import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Subset

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.dataset import create_dataset
from src.models.agent import Agent
from src.models.base_model import BaseModelWrapper
from src.utils.answer_extraction import extract_answer


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def build_default_output_path(output_dir: Path, model_name: str, max_samples: int | None) -> Path:
    slug = model_name.split("/")[-1].lower()
    sample_tag = "all" if max_samples is None or max_samples < 0 else str(max_samples)
    return output_dir / f"single_model_{slug}_{sample_tag}.json"


def build_generation_metadata(generated_token_ids: list[int], eos_token_id: int | None, max_new_tokens: int) -> dict:
    finish_reason = Agent._infer_finish_reason(
        generated_ids=generated_token_ids,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    return {
        "finish_reason": finish_reason,
        "generated_token_count": len(generated_token_ids),
        "stopped_early": finish_reason != "max_new_tokens",
    }


def setup_distributed() -> tuple[torch.device, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return device, rank, world_size, True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def shard_dataset(dataset, rank: int, world_size: int):
    if world_size <= 1:
        return dataset
    return Subset(dataset, list(range(rank, len(dataset), world_size)))


def gather_sharded_objects(local_obj, world_size: int):
    if world_size <= 1:
        return [local_obj]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_obj)
    return gathered


def run_single_model_baseline(
    model_name: str,
    max_samples: int | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    batch_size: int = 1,
) -> dict:
    if max_samples is not None and max_samples < 0:
        max_samples = None

    device, rank, world_size, is_dist = setup_distributed()
    model = BaseModelWrapper(model_name=model_name)
    model.tokenizer.padding_side = "left"
    model.to(device)
    model.eval()

    full_dataset = create_dataset(task="gsm8k", split="test", max_samples=max_samples)
    dataset = shard_dataset(full_dataset, rank, world_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    correct = 0
    total = 0
    samples = []

    with torch.no_grad():
        for batch in dataloader:
            tokenized = model.tokenize(batch["questions"], max_length=256)
            task_ids = tokenized["input_ids"].to(device)
            task_mask = tokenized["attention_mask"].to(device)

            gen_out = model.model.generate(
                input_ids=task_ids,
                attention_mask=task_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=model.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
            sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out[0]

            for i, question in enumerate(batch["questions"]):
                prompt_len = int(task_mask[i].sum().item())
                generated_token_ids = sequences[i][prompt_len:].tolist()
                text = model.tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=True,
                )
                generation = build_generation_metadata(
                    generated_token_ids=generated_token_ids,
                    eos_token_id=model.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                )

                pred = extract_answer(text, task_type="gsm8k")
                gold = batch["answers"][i].strip()
                ok = pred.strip() == gold.strip()
                correct += int(ok)
                total += 1
                samples.append(
                    {
                        "question": question,
                        "gold": gold,
                        "prediction": pred,
                        "generated_text": text[:500],
                        "generation": generation,
                        "correct": ok,
                    }
                )

    accuracy = correct / total * 100 if total > 0 else 0.0
    local_result = {
        "method": "single_model",
        "task": "gsm8k",
        "metrics": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
        },
        "parameters": {
            "model_name": model_name,
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "batch_size": batch_size,
            "world_size": world_size,
        },
        "samples": samples,
    }
    gathered = gather_sharded_objects(local_result, world_size)
    if not is_main_process(rank):
        if is_dist:
            dist.barrier()
            cleanup_distributed()
        return {}

    merged_samples = []
    total = 0
    correct = 0
    for shard in gathered:
        total += shard["metrics"]["total"]
        correct += shard["metrics"]["correct"]
        merged_samples.extend(shard["samples"])
    merged = {
        "method": "single_model",
        "task": "gsm8k",
        "metrics": {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total * 100) if total > 0 else 0.0,
        },
        "parameters": {
            "model_name": model_name,
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "batch_size": batch_size,
            "world_size": world_size,
        },
        "samples": merged_samples,
    }
    if is_dist:
        dist.barrier()
    cleanup_distributed()
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else build_default_output_path(
        output_dir=output_dir,
        model_name=args.model_name,
        max_samples=args.max_samples,
    )

    result = run_single_model_baseline(
        model_name=args.model_name,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        batch_size=args.batch_size,
    )
    if not result:
        return

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
