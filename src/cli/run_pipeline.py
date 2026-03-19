import argparse
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_DIR = PROJECT_ROOT / "src" / "cli"
UV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def build_runner(nproc_per_node: int) -> list[str]:
    if nproc_per_node > 1:
        return ["uv", "run", "--python", UV_PYTHON, "torchrun", "--master_port=29611", f"--nproc_per_node={nproc_per_node}"]
    return ["uv", "run", "--python", UV_PYTHON]


def run_command(cmd: list[str], env_overrides: dict[str, str] | None = None):
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def common_env(args) -> dict[str, str]:
    env = {}
    if args.hf_home:
        env["HF_HOME"] = args.hf_home
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def main():
    parser = argparse.ArgumentParser(description="Unified launcher for training and evaluation")
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--offline", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--config", required=True)
    train_p.add_argument("--max-samples", type=int, default=None)
    train_p.add_argument("--nproc-per-node", type=int, default=1)

    ours_p = subparsers.add_parser("eval-ours")
    ours_p.add_argument("--config", required=True)
    ours_p.add_argument("--checkpoint", required=True)
    ours_p.add_argument("--max-samples", type=int, default=None)
    ours_p.add_argument("--max-new-tokens", type=int, default=2048)
    ours_p.add_argument(
        "--inference-mode",
        type=str,
        default="chat_with_prefix",
        choices=["chat_with_prefix", "legacy_plain_with_prefix"],
    )
    ours_p.add_argument("--no-terminal-prefix", action="store_true")
    ours_p.add_argument("--do-sample", action="store_true")
    ours_p.add_argument("--run-baseline", action="store_true")
    ours_p.add_argument("--nproc-per-node", type=int, default=1)

    single_p = subparsers.add_parser("eval-single")
    single_p.add_argument("--model-name", required=True)
    single_p.add_argument("--max-samples", type=int, default=16)
    single_p.add_argument("--max-new-tokens", type=int, default=2048)
    single_p.add_argument("--batch-size", type=int, default=1)
    single_p.add_argument("--do-sample", action="store_true")
    single_p.add_argument("--output", default=None)
    single_p.add_argument("--output-dir", default="outputs/baselines")
    single_p.add_argument("--nproc-per-node", type=int, default=1)

    paper_p = subparsers.add_parser("eval-paper")
    paper_p.add_argument("--model-name", required=True)
    paper_p.add_argument("--max-samples", type=int, default=16)
    paper_p.add_argument("--latent-steps", type=int, default=10)
    paper_p.add_argument("--max-new-tokens", type=int, default=2048)
    paper_p.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    paper_p.add_argument("--output", default=None)
    paper_p.add_argument("--output-dir", default="outputs/baselines")
    paper_p.add_argument("--nproc-per-node", type=int, default=1)

    args = parser.parse_args()
    env = common_env(args)

    if args.command == "train":
        cmd = build_runner(args.nproc_per_node) + [str(CLI_DIR / ("multi_train.py" if args.nproc_per_node > 1 else "train.py")), "--config", args.config]
        if args.max_samples is not None:
            cmd.extend(["--max_samples", str(args.max_samples)])
        run_command(cmd, env)
        return

    if args.command == "eval-ours":
        cmd = build_runner(args.nproc_per_node) + [
            str(CLI_DIR / "evaluate.py"),
            "--config", args.config,
            "--checkpoint", args.checkpoint,
            "--max-new-tokens", str(args.max_new_tokens),
            "--inference-mode", args.inference_mode,
        ]
        if args.max_samples is not None:
            cmd.extend(["--max_samples", str(args.max_samples)])
        if args.no_terminal_prefix:
            cmd.append("--no-terminal-prefix")
        if args.do_sample:
            cmd.append("--do-sample")
        if args.run_baseline:
            cmd.append("--run-baseline")
        run_command(cmd, env)
        return

    if args.command == "eval-single":
        cmd = build_runner(args.nproc_per_node) + [
            str(CLI_DIR / "run_baseline_single_model.py"),
            "--model-name", args.model_name,
            "--max-samples", str(args.max_samples),
            "--max-new-tokens", str(args.max_new_tokens),
            "--batch-size", str(args.batch_size),
            "--output-dir", args.output_dir,
        ]
        if args.do_sample:
            cmd.append("--do-sample")
        if args.output:
            cmd.extend(["--output", args.output])
        run_command(cmd, env)
        return

    if args.command == "eval-paper":
        cmd = build_runner(args.nproc_per_node) + [
            str(CLI_DIR / "run_baseline_paper_latentmas.py"),
            "--model-name", args.model_name,
            "--max-samples", str(args.max_samples),
            "--latent-steps", str(args.latent_steps),
            "--max-new-tokens", str(args.max_new_tokens),
            "--prompt", args.prompt,
            "--output-dir", args.output_dir,
        ]
        if args.output:
            cmd.extend(["--output", args.output])
        run_command(cmd, env)
        return


if __name__ == "__main__":
    main()
