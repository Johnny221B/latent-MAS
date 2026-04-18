"""
Standard SFT baseline using HuggingFace Trainer.
Usage:
    python scripts/sft_baseline.py \
        --model Qwen/Qwen3-4B \
        --dataset am_deepseek_r1_distilled \
        --source limo \
        --output_dir outputs/rq3/single_sft_hf \
        --lr 1e-5 --epochs 2 --batch_size 2
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from src.data import create_dataset


def format_chat(tokenizer, question: str, answer: str, max_len: int = 16384):
    """Format as chat template: system + user question → assistant answer."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You must reason step-by-step to solve the provided question.\nYour output MUST follow this format: first show your reasoning inside <think> </think> tags, then provide the final answer inside <answer> </answer> tags.\nExample: <think> step-by-step reasoning here </think><answer> final answer here </answer>"},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer + tokenizer.eos_token, add_special_tokens=False)

    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="am_deepseek_r1_distilled")
    parser.add_argument("--source", type=str, default="limo")
    parser.add_argument("--output_dir", type=str, default="outputs/rq3/single_sft_hf")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=16384)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lora", action="store_true", help="Use LoRA instead of full fine-tune")
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print(f"Loading dataset: {args.dataset} (source={args.source})")
    raw_dataset = create_dataset(task=args.dataset, split="train", source=args.source)
    print(f"Raw samples: {len(raw_dataset)}")

    processed = []
    skipped = 0
    for i in range(len(raw_dataset)):
        item = raw_dataset[i]
        result = format_chat(tokenizer, item["question"], item["answer"], max_len=args.max_len)
        if len(result["input_ids"]) < 10:
            skipped += 1
            continue
        processed.append(result)
    print(f"Processed: {len(processed)}, skipped: {skipped}")

    train_dataset = Dataset.from_list(processed)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=f"sft_baseline_{args.source}_{Path(args.model).name}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
