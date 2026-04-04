# import argparse
# from pathlib import Path

# import matplotlib.pyplot as plt
# import pandas as pd


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True, help="Path to loss_log.csv")
#     parser.add_argument("--output", default=None, help="Output image path")
#     args = parser.parse_args()

#     input_path = Path(args.input)

#     if args.output is None:
#         output_path = input_path.parent / "epoch_loss_curve.png"
#     else:
#         output_path = Path(args.output)

#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(input_path)
#     epoch_loss = df.groupby("epoch")[["loss", "task_loss", "graph_loss"]].mean().reset_index()

#     plt.figure(figsize=(8, 5))
#     plt.plot(epoch_loss["epoch"], epoch_loss["loss"], marker="o", label="loss")
#     plt.plot(epoch_loss["epoch"], epoch_loss["task_loss"], marker="o", label="task_loss")
#     plt.plot(epoch_loss["epoch"], epoch_loss["graph_loss"], marker="o", label="graph_loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Loss Curves per Epoch")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     print(f"Saved plot to {output_path}")


# if __name__ == "__main__":
#     main()

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_from_log(log_path, window=20, save_dir=None):
    """
    解析训练日志中的 loss，并绘制：
    1. batch-level loss + moving average
    2. epoch-level average loss

    参数
    ----
    log_path : str
        output.log 文件路径
    window : int
        batch loss 滑动平均窗口大小
    save_dir : str or None
        图片保存目录。None 表示默认保存在 log 文件同目录下

    日志示例
    --------
    E1 B1/233 | Loss:13.0465 Task:13.0465 Graph:0.0000 | ...
    """

    log_path = Path(log_path)
    if save_dir is None:
        save_dir = log_path.parent
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 匹配 epoch, batch, total_batches, loss
    pattern = re.compile(
        r"E(?P<epoch>\d+)\s+B(?P<batch>\d+)/(?P<total_batches>\d+)\s+\|\s+Loss:(?P<loss>[-+]?\d*\.?\d+)"
    )

    records = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch = int(m.group("epoch"))
                batch = int(m.group("batch"))
                total_batches = int(m.group("total_batches"))
                loss = float(m.group("loss"))

                global_step = (epoch - 1) * total_batches + batch

                records.append(
                    {
                        "epoch": epoch,
                        "batch": batch,
                        "total_batches": total_batches,
                        "global_step": global_step,
                        "loss": loss,
                    }
                )

    if not records:
        raise ValueError(f"没有在日志中解析到 loss 记录，请检查文件格式：{log_path}")

    df = pd.DataFrame(records).sort_values(["epoch", "batch"]).reset_index(drop=True)

    # batch loss 的滑动平均
    df["loss_smooth"] = df["loss"].rolling(window=window, min_periods=1).mean()

    # 每个 epoch 的平均 loss
    epoch_df = (
        df.groupby("epoch", as_index=False)["loss"]
        .mean()
        .rename(columns={"loss": "epoch_avg_loss"})
    )

    # ===== 图1：batch-level loss + moving average =====
    plt.figure(figsize=(10, 5))
    plt.plot(df["global_step"], df["loss"], alpha=0.35, label="Batch Loss")
    plt.plot(df["global_step"], df["loss_smooth"], linewidth=2, label=f"Moving Avg (window={window})")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Batch Loss and Moving Average")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    batch_fig_path = save_dir / "batch_loss_with_moving_avg.png"
    plt.savefig(batch_fig_path, dpi=200)
    plt.show()
    plt.close()

    # ===== 图2：epoch-level average loss =====
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_df["epoch"], epoch_df["epoch_avg_loss"], marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Epoch Average Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    epoch_fig_path = save_dir / "epoch_avg_loss.png"
    plt.savefig(epoch_fig_path, dpi=200)
    plt.show()
    plt.close()

    print(f"共解析到 {len(df)} 条 batch loss 记录")
    print(f"batch 曲线已保存到: {batch_fig_path}")
    print(f"epoch 曲线已保存到: {epoch_fig_path}")

    return df, epoch_df


if __name__ == "__main__":
    # 这里改成你的 output.log 路径
    log_path = "/home/chengzhi.ucsb/code/toby/latent-MAS/outputs/gsm8k_qwen3-4b_20260402_172227/wandb/latest-run/files/output.log"

    # window 可以改，比如 10 / 20 / 50
    df, epoch_df = plot_loss_from_log(log_path, window=20)