import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.dataset import create_dataset
from src.utils.answer_extraction import extract_answer


def build_default_output_path(output_dir: Path, model_name: str, max_samples: int | None) -> Path:
    slug = model_name.split("/")[-1].lower()
    sample_tag = "all" if max_samples is None or max_samples < 0 else str(max_samples)
    return output_dir / f"single_model_{slug}_{sample_tag}.json"


def build_qwen_chat_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": question}]


def build_chat_payload(
    model_name: str,
    question: str,
    max_new_tokens: int,
    do_sample: bool,
) -> dict:
    payload = {
        "model": model_name,
        "messages": build_qwen_chat_messages(question),
        "max_tokens": max_new_tokens,
    }
    if do_sample:
        payload["temperature"] = 0.7
        payload["top_p"] = 0.8
    else:
        payload["temperature"] = 0.0
    return payload


def build_generation_metadata(choice: dict, max_new_tokens: int, content: str) -> dict:
    finish_reason = choice.get("finish_reason", "unknown")
    usage = choice.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len(content.split())
    return {
        "finish_reason": finish_reason,
        "generated_token_count": completion_tokens,
        "stopped_early": finish_reason != "length" and completion_tokens < max_new_tokens,
    }


def launch_vllm_server(
    model_name: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    api_key: str,
) -> subprocess.Popen:
    python_bin = project_root / ".venv" / "bin" / "python"
    cmd = [
        str(python_bin),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--api-key",
        api_key,
    ]
    print("Starting vLLM server:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


def stop_vllm_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_for_vllm_ready(base_url: str, api_key: str, timeout_seconds: int = 600) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=10)
            if response.ok:
                return
            last_error = f"{response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"vLLM server did not become ready within {timeout_seconds}s: {last_error}")


def request_completion(
    session: requests.Session,
    base_url: str,
    api_key: str,
    payload: dict,
) -> dict:
    response = session.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def evaluate_one(
    index: int,
    sample: dict,
    model_name: str,
    max_new_tokens: int,
    do_sample: bool,
    base_url: str,
    api_key: str,
) -> dict:
    payload = build_chat_payload(
        model_name=model_name,
        question=sample["question"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    with requests.Session() as session:
        response = request_completion(session, base_url=base_url, api_key=api_key, payload=payload)
    choice = response["choices"][0]
    message = choice.get("message", {})
    content = message.get("content", "")
    prediction = extract_answer(content, task_type="gsm8k")
    gold = sample["answer"].strip()
    correct = prediction.strip() == gold.strip()
    generation = build_generation_metadata(
        choice={
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
        },
        max_new_tokens=max_new_tokens,
        content=content,
    )
    return {
        "index": index,
        "question": sample["question"],
        "gold": gold,
        "prediction": prediction,
        "generated_text": content,
        "generated_text_preview": content[:500],
        "generation": generation,
        "correct": correct,
    }


def run_single_model_baseline(
    model_name: str,
    max_samples: int | None,
    max_new_tokens: int,
    do_sample: bool,
    worker_count: int,
    base_url: str,
    api_key: str,
) -> dict:
    if max_samples is not None and max_samples < 0:
        max_samples = None

    dataset = create_dataset(task="gsm8k", split="test", max_samples=max_samples)
    indexed_samples = list(enumerate(dataset))

    results_by_index: dict[int, dict] = {}
    correct = 0
    total = 0
    normal_stop_count = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                evaluate_one,
                index=index,
                sample=sample,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                base_url=base_url,
                api_key=api_key,
            )
            for index, sample in indexed_samples
        ]
        for future in as_completed(futures):
            result = future.result()
            idx = result.pop("index")
            results_by_index[idx] = result
            correct += int(result["correct"])
            total += 1
            normal_stop_count += int(result["generation"]["finish_reason"] == "stop")
            if total % 10 == 0 or total == len(indexed_samples):
                accuracy = correct / total * 100 if total > 0 else 0.0
                print(f"[{total}/{len(indexed_samples)}] Acc: {accuracy:.2f}% ({correct}/{total})")

    samples = [results_by_index[idx] for idx in sorted(results_by_index)]
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"Normal stop datapoints: {normal_stop_count}/{total}")
    return {
        "method": "single_model_vllm_api",
        "task": "gsm8k",
        "metrics": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "normal_stop_count": normal_stop_count,
            "normal_stop_rate": (normal_stop_count / total * 100) if total > 0 else 0.0,
        },
        "parameters": {
            "model_name": model_name,
            "prompt_format": "openai_chat_api",
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "worker_count": worker_count,
            "base_url": base_url,
        },
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--output", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reuse-server", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else build_default_output_path(
        output_dir=output_dir,
        model_name=args.model_name,
        max_samples=args.max_samples,
    )

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    server_proc = None
    if not args.reuse_server:
        server_proc = launch_vllm_server(
            model_name=args.model_name,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            api_key=args.api_key,
        )
        atexit.register(stop_vllm_server, server_proc)

    try:
        wait_for_vllm_ready(base_url=base_url, api_key=args.api_key)
        result = run_single_model_baseline(
            model_name=args.model_name,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            worker_count=args.worker_count,
            base_url=base_url,
            api_key=args.api_key,
        )
        result["parameters"].update(
            {
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "reuse_server": args.reuse_server,
            }
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output_path}")
    finally:
        stop_vllm_server(server_proc)


if __name__ == "__main__":
    main()
