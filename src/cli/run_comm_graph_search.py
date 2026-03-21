"""Automatic communication-only graph search for probe64 experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.cli.train import train
from src.utils.config import load_config


@dataclass
class ExperimentSpec:
    iteration: int
    phase: str
    slug: str
    description: str
    overrides: dict


@dataclass
class EvalWatchdogState:
    eval_started_at: float | None = None
    last_reason: str | None = None


def build_stage1_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(1, "stage1", "baseline", "Current communication-only baseline with campaign invariants.", {}),
        ExperimentSpec(
            2,
            "stage1",
            "lambda-rebalance",
            "Raise lambda_drop so prior-edge graph gradients no longer cancel.",
            {"training": {"lambda_drop": 0.02, "lambda_sparse": 0.005}},
        ),
        ExperimentSpec(
            3,
            "stage1",
            "low-init-scale",
            "Reduce adjacency init saturation.",
            {"graph": {"init_scale": 1.5}},
        ),
        ExperimentSpec(
            4,
            "stage1",
            "lambda-plus-low-init",
            "Combine graph regularizer rebalance with lower init saturation.",
            {
                "training": {"lambda_drop": 0.02, "lambda_sparse": 0.005},
                "graph": {"init_scale": 1.5},
            },
        ),
        ExperimentSpec(
            5,
            "stage1",
            "prior-residual",
            "Use prior-residual graph parameterization with lower init saturation.",
            {"graph": {"parameterization": "prior_residual", "init_scale": 1.5}},
        ),
    ]


def build_stage2_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(6, "stage2", "longer-schedule", "Double epochs while keeping backbone frozen.", {"training": {"epochs": 128}}),
        ExperimentSpec(
            7,
            "stage2",
            "bigger-prefix",
            "Increase communication capacity through more compressor queries.",
            {"compressor": {"num_queries": 32}},
        ),
        ExperimentSpec(
            8,
            "stage2",
            "longer-rollout",
            "Increase latent rollout without enabling thinking or backbone finetuning.",
            {"reasoning": {"steps_per_agent": 40, "compress_last_k": 40}},
        ),
    ]


def _deep_merge(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def prepare_iteration_config(base_config: dict, spec: ExperimentSpec, campaign_root: Path) -> dict:
    config = copy.deepcopy(base_config)
    _deep_merge(config, spec.overrides)

    training = config.setdefault("training", {})
    evaluation = config.setdefault("evaluation", {})
    graph = config.setdefault("graph", {})
    output = config.setdefault("output", {})

    training["train_strategy"] = "communication_only"
    training["batch_size"] = 32
    training["save_final_checkpoint"] = False
    evaluation["run_after_train"] = True
    evaluation["train_probe_samples"] = 0
    evaluation["test_probe_samples"] = 64
    evaluation["write_agent_logs"] = True
    evaluation.setdefault("batch_size", 32)
    evaluation["max_new_tokens"] = 12000
    evaluation["inference_mode"] = "chat_with_prefix"
    evaluation["use_terminal_prefix"] = True
    evaluation["do_sample"] = False
    output["dir"] = str(campaign_root / f"iter{spec.iteration:02d}_{spec.slug}")
    graph.setdefault("parameterization", "sigmoid_logits")
    graph.setdefault("init_scale", 5.0)

    return config


def _write_leaderboard(campaign_root: Path, results: list[dict]) -> None:
    leaderboard_path = campaign_root / "leaderboard.json"
    summary_path = campaign_root / "summary.md"
    leaderboard_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Communication Graph Search Summary",
        "",
        "| Iter | Phase | Slug | Status | Train Acc | Test Acc | Output Dir |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['iteration']} | {row['phase']} | {row['slug']} | "
            f"{row.get('status', 'completed')} | {row['train_accuracy']:.2f} | {row['test_accuracy']:.2f} | {row['output_dir']} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_train_command(
    config_path: Path,
    max_samples: int,
    nproc_per_node: int,
    python_bin: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "--python",
        python_bin,
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "src/cli/train.py",
        "--config",
        str(config_path),
        "--max_samples",
        str(max_samples),
    ]


def discover_python_bin(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path
    candidates = [project_root / ".venv/bin/python"]
    candidates.extend(parent / ".venv/bin/python" for parent in project_root.parents)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(Path(sys.executable).resolve())


def discover_hf_home(explicit_path: str | None = None) -> str | None:
    if explicit_path:
        return explicit_path
    candidates = [project_root / ".cache/huggingface"]
    candidates.extend(parent / ".cache/huggingface" for parent in project_root.parents)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def find_iteration_output_dir(base_output_dir: Path, existing_paths: set[Path]) -> Path:
    parent = base_output_dir.parent
    pattern = f"{base_output_dir.name}_*"
    candidates = {path.resolve() for path in parent.glob(pattern) if path.is_dir()}
    new_candidates = sorted(candidates - existing_paths)
    if new_candidates:
        return new_candidates[-1]
    if candidates:
        return sorted(candidates)[-1]
    raise FileNotFoundError(f"No timestamped output directory found for {base_output_dir}")


def load_eval_summary(output_dir: Path, filename: str) -> dict:
    path = output_dir / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_eval_progress_log(log_text: str) -> str | None:
    if "[WARN]" in log_text or "suspected gibberish" in log_text:
        return "eval preview flagged suspected gibberish"

    suspicious_finish = "Finish: max_new_tokens | Tokens: 12000" in log_text
    suspicious_gen = any(marker in log_text for marker in ["Gen: .", "Gen: \t.", "Gen: <empty>", "Pred: <empty>"])
    if suspicious_finish and suspicious_gen:
        return "eval preview shows max_new_tokens saturation with near-empty generations"
    return None


def maybe_abort_for_eval_gibberish(
    progress_log: Path,
    state: EvalWatchdogState,
    now: float,
    threshold_seconds: float = 1800.0,
) -> str | None:
    if not progress_log.exists():
        return None

    log_text = progress_log.read_text(encoding="utf-8")
    if "Running evaluation..." not in log_text:
        return None

    if state.eval_started_at is None:
        state.eval_started_at = now
        return None

    if (now - state.eval_started_at) < threshold_seconds:
        return None

    reason = analyze_eval_progress_log(log_text)
    if reason is None:
        return None
    state.last_reason = reason
    return f"Watchdog aborted iteration after {int(now - state.eval_started_at)}s of eval: gibberish detected ({reason})"


def run_training_iteration(
    iter_config_path: Path,
    iter_config: dict,
    max_samples: int,
    launcher: str,
    nproc_per_node: int,
    python_bin: str,
    hf_home: str | None,
    offline: bool,
    cuda_visible_devices: str | None,
) -> dict:
    base_output_dir = Path(iter_config["output"]["dir"])
    existing_paths = {
        path.resolve()
        for path in base_output_dir.parent.glob(f"{base_output_dir.name}_*")
        if path.is_dir()
    }

    if launcher == "direct":
        result = train(str(iter_config_path), max_samples=max_samples)
        output_dir = Path(result["output_dir"])
        eval_summaries = result.get("eval_summaries") or {}
        return {
            "output_dir": str(output_dir),
            "train_summary": eval_summaries.get("train", {}),
            "test_summary": eval_summaries.get("test", {}),
        }

    command = build_train_command(
        config_path=iter_config_path,
        max_samples=max_samples,
        nproc_per_node=nproc_per_node,
        python_bin=python_bin,
    )
    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = hf_home
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    process = subprocess.Popen(command, cwd=project_root, env=env)
    watchdog_state = EvalWatchdogState()
    output_dir: Path | None = None
    progress_log: Path | None = None

    while True:
        return_code = process.poll()
        if output_dir is None:
            try:
                output_dir = find_iteration_output_dir(base_output_dir, existing_paths)
                progress_log = output_dir / "eval_progress.log"
            except FileNotFoundError:
                output_dir = None
                progress_log = None

        if return_code is not None:
            if return_code != 0:
                raise subprocess.CalledProcessError(returncode=return_code, cmd=command)
            break

        if progress_log is not None:
            abort_reason = maybe_abort_for_eval_gibberish(
                progress_log=progress_log,
                state=watchdog_state,
                now=time.time(),
            )
            if abort_reason is not None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                raise RuntimeError(abort_reason)

        time.sleep(60)

    if output_dir is None:
        output_dir = find_iteration_output_dir(base_output_dir, existing_paths)
    return {
        "output_dir": str(output_dir),
        "train_summary": load_eval_summary(output_dir, "eval_results_train.json"),
        "test_summary": load_eval_summary(output_dir, "eval_results.json"),
    }


def run_search(
    base_config_path: str,
    campaign_root: str,
    target_accuracy: float = 80.0,
    min_iterations: int = 5,
    max_samples: int = 64,
    launcher: str = "torchrun",
    nproc_per_node: int = 2,
    python_bin: str = ".venv/bin/python",
    hf_home: str | None = None,
    offline: bool = True,
    cuda_visible_devices: str | None = None,
) -> dict:
    campaign_root_path = Path(campaign_root)
    campaign_root_path.mkdir(parents=True, exist_ok=True)
    base_config = load_config(base_config_path)
    results: list[dict] = []
    best_config = copy.deepcopy(base_config)
    best_test = float("-inf")

    for spec in build_stage1_specs():
        config = prepare_iteration_config(best_config if spec.phase == "stage2" else base_config, spec, campaign_root_path)
        config_path = campaign_root_path / "configs"
        config_path.mkdir(parents=True, exist_ok=True)
        iter_config_path = config_path / f"iter{spec.iteration:02d}_{spec.slug}.yaml"
        iter_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        try:
            result = run_training_iteration(
                iter_config_path=iter_config_path,
                iter_config=config,
                max_samples=max_samples,
                launcher=launcher,
                nproc_per_node=nproc_per_node,
                python_bin=python_bin,
                hf_home=hf_home,
                offline=offline,
                cuda_visible_devices=cuda_visible_devices,
            )
            train_acc = result["train_summary"].get("metrics", {}).get("accuracy", 0.0)
            test_acc = result["test_summary"].get("metrics", {}).get("accuracy", 0.0)
            entry = {
                "iteration": spec.iteration,
                "phase": spec.phase,
                "slug": spec.slug,
                "description": spec.description,
                "status": "completed",
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "output_dir": result["output_dir"],
                "config_path": str(iter_config_path),
            }
        except Exception as exc:
            entry = {
                "iteration": spec.iteration,
                "phase": spec.phase,
                "slug": spec.slug,
                "description": spec.description,
                "status": "failed",
                "train_accuracy": 0.0,
                "test_accuracy": 0.0,
                "output_dir": "",
                "config_path": str(iter_config_path),
                "error": str(exc),
            }
            results.append(entry)
            _write_leaderboard(campaign_root_path, results)
            continue
        results.append(entry)
        _write_leaderboard(campaign_root_path, results)

        if test_acc > best_test:
            best_test = test_acc
            best_config = copy.deepcopy(config)

    if best_test < target_accuracy:
        for spec in build_stage2_specs():
            config = prepare_iteration_config(best_config, spec, campaign_root_path)
            config_path = campaign_root_path / "configs"
            iter_config_path = config_path / f"iter{spec.iteration:02d}_{spec.slug}.yaml"
            iter_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            try:
                result = run_training_iteration(
                    iter_config_path=iter_config_path,
                    iter_config=config,
                    max_samples=max_samples,
                    launcher=launcher,
                    nproc_per_node=nproc_per_node,
                    python_bin=python_bin,
                    hf_home=hf_home,
                    offline=offline,
                    cuda_visible_devices=cuda_visible_devices,
                )
                train_acc = result["train_summary"].get("metrics", {}).get("accuracy", 0.0)
                test_acc = result["test_summary"].get("metrics", {}).get("accuracy", 0.0)
                entry = {
                    "iteration": spec.iteration,
                    "phase": spec.phase,
                    "slug": spec.slug,
                    "description": spec.description,
                    "status": "completed",
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "output_dir": result["output_dir"],
                    "config_path": str(iter_config_path),
                }
            except Exception as exc:
                entry = {
                    "iteration": spec.iteration,
                    "phase": spec.phase,
                    "slug": spec.slug,
                    "description": spec.description,
                    "status": "failed",
                    "train_accuracy": 0.0,
                    "test_accuracy": 0.0,
                    "output_dir": "",
                    "config_path": str(iter_config_path),
                    "error": str(exc),
                }
                results.append(entry)
                _write_leaderboard(campaign_root_path, results)
                continue
            results.append(entry)
            _write_leaderboard(campaign_root_path, results)

            if test_acc > best_test:
                best_test = test_acc
                best_config = copy.deepcopy(config)

            if len(results) >= min_iterations and best_test >= target_accuracy:
                break

    return {
        "campaign_root": str(campaign_root_path),
        "best_test_accuracy": best_test if best_test != float("-inf") else 0.0,
        "iterations": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/experiments/gsm8k_5agent_probe64_comm_only.yaml")
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--target-accuracy", type=float, default=80.0)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--launcher", choices=["direct", "torchrun"], default="torchrun")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--python-bin")
    parser.add_argument("--hf-home")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    run_search(
        base_config_path=args.base_config,
        campaign_root=args.campaign_root,
        target_accuracy=args.target_accuracy,
        min_iterations=args.min_iterations,
        max_samples=args.max_samples,
        launcher=args.launcher,
        nproc_per_node=args.nproc_per_node,
        python_bin=discover_python_bin(args.python_bin),
        hf_home=discover_hf_home(args.hf_home),
        offline=not args.online,
        cuda_visible_devices=args.cuda_visible_devices,
    )


if __name__ == "__main__":
    main()
