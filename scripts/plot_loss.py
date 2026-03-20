import pandas as pd
import matplotlib.pyplot as plt

# 读取 csv
df = pd.read_csv("/data2/yangyz/latent-MAS/outputs/gsm8k_qwen3-0.6b_20260318_114558/loss_log.csv")

# 按 epoch 聚合，计算每个 epoch 的平均 loss
epoch_loss = df.groupby("epoch")[["loss", "task_loss", "graph_loss"]].mean().reset_index()

# 画图
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
plt.show()

plt.savefig("epoch_loss_curve.png", dpi=300, bbox_inches="tight")