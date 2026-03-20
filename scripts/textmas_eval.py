"""
TextMAS Baseline: Multi-agent system with text-based communication.

Each agent generates text normally, passes it as context to downstream agents.
No compressor, no latent space, no training. Pure text-based multi-agent pipeline.

This is the standard multi-agent baseline to compare against our latent communication system.

Usage:
    # GSM8K with Qwen3-0.6B
    CUDA_VISIBLE_DEVICES=0 python scripts/textmas_eval.py \
        --model_path /data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B \
        --task gsm8k \
        --max_samples 50

    # GSM8K with Qwen3-4B, full test set
    CUDA_VISIBLE_DEVICES=0 python scripts/textmas_eval.py \
        --model_path /data2/yangyz/latent-MAS/weights/Qwen__Qwen3-4B \
        --task gsm8k

    # Custom graph (3-agent sequential)
    CUDA_VISIBLE_DEVICES=0 python scripts/textmas_eval.py \
        --model_path /data2/yangyz/latent-MAS/weights/Qwen__Qwen3-0.6B \
        --task gsm8k \
        --graph sequential
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


# ── Agent definitions ──
# Each agent has a role name and a prompt template.
# {question} = the original question
# {context} = text output from upstream agents

GRAPH_CONFIGS = {
    "two_path": {
        "description": "reader->planner->solver->summarizer + reader->analyst->summarizer",
        "agents": [
            {
                "name": "reader",
                "upstream": [],
                "prompt": (
                    "You are a reader agent. Carefully read the problem and identify "
                    "key information, numbers, and what is being asked.\n\n"
                    "Problem: {question}\n\n"
                    "Your analysis:"
                ),
            },
            {
                "name": "planner",
                "upstream": ["reader"],
                "prompt": (
                    "You are a planning agent. Based on the reader's analysis, "
                    "break down the problem into clear solution steps.\n\n"
                    "Problem: {question}\n\n"
                    "Reader's analysis:\n{reader}\n\n"
                    "Your plan:"
                ),
            },
            {
                "name": "analyst",
                "upstream": ["reader"],
                "prompt": (
                    "You are an analyst agent. Based on the reader's analysis, "
                    "identify the type of problem and suggest solution strategies.\n\n"
                    "Problem: {question}\n\n"
                    "Reader's analysis:\n{reader}\n\n"
                    "Your analysis:"
                ),
            },
            {
                "name": "solver",
                "upstream": ["planner"],
                "prompt": (
                    "You are a solver agent. Follow the plan and execute the solution "
                    "step by step. Show all calculations.\n\n"
                    "Problem: {question}\n\n"
                    "Plan:\n{planner}\n\n"
                    "Your solution:"
                ),
            },
            {
                "name": "summarizer",
                "upstream": ["solver", "analyst"],
                "is_terminal": True,
                "prompt": (
                    "You are a summarizer agent. Review the solver's work and the analyst's "
                    "insights, then produce the final answer.\n\n"
                    "Problem: {question}\n\n"
                    "Solver's work:\n{solver}\n\n"
                    "Analyst's insights:\n{analyst}\n\n"
                    "Give the final answer. For math problems, put the numeric answer after ####.\n\n"
                    "Final answer:"
                ),
            },
        ],
    },
    "sequential": {
        "description": "planner->critic->refiner->solver (like LatentMAS paper)",
        "agents": [
            {
                "name": "planner",
                "upstream": [],
                "prompt": (
                    "You are a Planner Agent. Design a clear, step-by-step plan "
                    "to solve the question. Do not produce the final answer.\n\n"
                    "Question: {question}\n\n"
                    "Your plan:"
                ),
            },
            {
                "name": "critic",
                "upstream": ["planner"],
                "prompt": (
                    "You are a Critic Agent. Evaluate the plan and provide feedback.\n\n"
                    "Question: {question}\n\n"
                    "Plan:\n{planner}\n\n"
                    "Your feedback:"
                ),
            },
            {
                "name": "refiner",
                "upstream": ["critic"],
                "prompt": (
                    "You are a Refiner Agent. Based on the feedback, provide an improved plan.\n\n"
                    "Question: {question}\n\n"
                    "Original plan and feedback:\n{critic}\n\n"
                    "Your refined plan:"
                ),
            },
            {
                "name": "solver",
                "upstream": ["refiner"],
                "is_terminal": True,
                "prompt": (
                    "You are a Solver Agent. Follow the refined plan and solve the problem.\n\n"
                    "Question: {question}\n\n"
                    "Refined plan:\n{refiner}\n\n"
                    "Solve step by step. Put the final numeric answer after ####.\n\n"
                    "Solution:"
                ),
            },
        ],
    },
}


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
) -> str:
    """Generate text from a prompt using the model."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
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

    generated_ids = gen_out[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_textmas_pipeline(
    model,
    tokenizer,
    question: str,
    device: torch.device,
    graph_config: dict,
    max_new_tokens_per_agent: int = 256,
    terminal_max_tokens: int = 256,
    temperature: float = 0.6,
) -> dict:
    """Run the full TextMAS pipeline on a single question.

    Returns:
        dict with agent_outputs, final_text, total_tokens
    """
    agents = graph_config["agents"]
    agent_outputs = {}  # name -> generated text
    total_tokens = 0

    for agent_cfg in agents:
        name = agent_cfg["name"]
        is_terminal = agent_cfg.get("is_terminal", False)

        # Build prompt: fill in {question} and any upstream agent outputs
        prompt = agent_cfg["prompt"].format(
            question=question,
            **agent_outputs,
        )

        # Generate
        max_tokens = terminal_max_tokens if is_terminal else max_new_tokens_per_agent
        output_text = generate_text(
            model, tokenizer, prompt, device,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        agent_outputs[name] = output_text
        total_tokens += len(tokenizer.encode(output_text))

    # Terminal agent's output is the final answer
    terminal_name = [a["name"] for a in agents if a.get("is_terminal", False)][0]
    final_text = agent_outputs[terminal_name]

    return {
        "agent_outputs": agent_outputs,
        "final_text": final_text,
        "total_tokens": total_tokens,
    }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


def evaluate(
    model_path: str,
    task: str,
    graph: str = "two_path",
    max_samples: int | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
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

    mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  GPU memory: {mem:.2f} GB")

    # ── Graph config ──
    graph_config = GRAPH_CONFIGS[graph]
    agent_names = [a["name"] for a in graph_config["agents"]]
    print(f"\nGraph: {graph} — {graph_config['description']}")
    print(f"  Agents: {' → '.join(agent_names)}")

    # ── Dataset ──
    print(f"\nLoading dataset: {task} (test)")
    dataset = create_dataset(task=task, split="test", max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"  Samples: {len(dataset)}")

    # ── Evaluate ──
    correct = 0
    total = 0
    total_tokens = 0
    results = []

    print(f"\nRunning TextMAS evaluation...")
    t_start = time.time()

    for idx, batch in enumerate(dataloader):
        t0 = time.time()

        question = batch["questions"][0]
        gold = batch["answers"][0].strip()

        # Run full pipeline
        pipeline_result = run_textmas_pipeline(
            model, tokenizer, question, device,
            graph_config=graph_config,
            max_new_tokens_per_agent=max_new_tokens,
            terminal_max_tokens=max_new_tokens,
            temperature=temperature,
        )

        final_text = pipeline_result["final_text"]
        total_tokens += pipeline_result["total_tokens"]

        # Extract answer
        pred = extract_answer(final_text, task_type=task).strip()
        is_correct = pred == gold

        if is_correct:
            correct += 1
        total += 1

        t1 = time.time()

        results.append({
            "question": question,
            "gold": gold,
            "prediction": pred,
            "final_text": final_text[:500],
            "agent_outputs": {k: v[:300] for k, v in pipeline_result["agent_outputs"].items()},
            "tokens": pipeline_result["total_tokens"],
            "correct": is_correct,
        })

        # Print examples
        if idx < 3:
            status = "✓" if is_correct else "✗"
            print(f"\n  Example {idx+1} [{status}]:")
            print(f"    Q: {question[:80]}...")
            print(f"    Gold: {gold}")
            print(f"    Pred: {pred}")
            for agent_name, agent_text in pipeline_result["agent_outputs"].items():
                print(f"    [{agent_name}]: {agent_text[:100]}...")

        # Progress
        if (idx + 1) % 10 == 0 or (idx + 1) == len(dataloader):
            acc = correct / total * 100
            elapsed = time.time() - t_start
            eta = elapsed / total * (len(dataloader) - total) if total < len(dataloader) else 0
            avg_tokens = total_tokens / total
            print(
                f"  [{idx+1}/{len(dataloader)}] "
                f"Acc: {acc:.1f}% ({correct}/{total}) | "
                f"Avg tokens: {avg_tokens:.0f} | "
                f"{t1-t0:.1f}s/sample | "
                f"ETA: {eta/60:.0f}min"
            )

    # ── Summary ──
    t_total = time.time() - t_start
    accuracy = correct / total * 100 if total > 0 else 0.0
    avg_tokens = total_tokens / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"  TEXTMAS RESULTS")
    print(f"{'='*60}")
    print(f"  Model:      {model_path}")
    print(f"  Task:       {task}")
    print(f"  Graph:      {graph} ({graph_config['description']})")
    print(f"  Samples:    {total}")
    print(f"  Correct:    {correct}")
    print(f"  Accuracy:   {accuracy:.2f}%")
    print(f"  Avg tokens: {avg_tokens:.0f} per question")
    print(f"  Time:       {t_total:.0f}s ({t_total/total:.1f}s/sample)")
    print(f"{'='*60}")

    # ── Save ──
    model_name = Path(model_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/textmas_{model_name}_{task}_{graph}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "model": model_path,
            "task": task,
            "graph": graph,
            "graph_description": graph_config["description"],
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_tokens": avg_tokens,
            "time_seconds": t_total,
            "temperature": temperature,
            "samples": results,
        }, f, indent=2, ensure_ascii=False)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Model:      {model_path}\n")
        f.write(f"Task:       {task}\n")
        f.write(f"Graph:      {graph}\n")
        f.write(f"Accuracy:   {accuracy:.2f}% ({correct}/{total})\n")
        f.write(f"Avg tokens: {avg_tokens:.0f}\n")
        f.write(f"Time:       {t_total:.0f}s\n")

    print(f"  Results saved: {output_dir}/")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextMAS baseline evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="gsm8k",
                        choices=["gsm8k", "arc_easy", "arc_challenge", "medqa", "math"])
    parser.add_argument("--graph", type=str, default="two_path",
                        choices=["two_path", "sequential"],
                        help="Agent graph topology")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        task=args.task,
        graph=args.graph,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )