import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to loss_log.csv")
    parser.add_argument("--output", default=None, help="Output image path")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output is None:
        output_path = input_path.parent / "epoch_loss_curve.png"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    epoch_loss = df.groupby("epoch")[["loss", "task_loss", "graph_loss"]].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_loss["epoch"], epoch_loss["loss"], marker="o", label="loss")
    plt.plot(epoch_loss["epoch"], epoch_loss["task_loss"], marker="o", label="task_loss")
    plt.plot(epoch_loss["epoch"], epoch_loss["graph_loss"], marker="o", label="graph_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves per Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()