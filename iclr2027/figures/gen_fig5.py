"""
Figure 5: Prediction results.
(a) Bar chart of within-task AUC vs baselines.
(b) Cross-task transfer heatmap.
(c) Cross-model transfer heatmap.
Generates: fig5_prediction.pdf / .png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
})

fig = plt.figure(figsize=(13.2, 4.6))
# 适当留宽间距，防止文字和 Colorbar 挤压
gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 0.95, 1.05], wspace=0.48)
fig.subplots_adjust(left=0.06, right=0.96, top=0.86, bottom=0.22)

# === (a) Within-task: full trajectory vs baselines ===
ax = fig.add_subplot(gs[0])
baselines = ["CIS\nslope", "Rank\n@t0", "Margin\n+slope", "Rank\nmin", "Max\n+slope", "Entropy", "Full\ntrajectory"]
aucs = [0.388, 0.511, 0.612, 0.486, 0.612, 0.325, 0.789]
colors = ["#95A5A6"]*6 + ["#C0392B"]
bars = ax.bar(baselines, aucs, color=colors, edgecolor="black", linewidth=0.7, width=0.68)
ax.set_ylim(0, 1.08)
ax.set_ylabel("AUC (within-task)")
ax.set_title("(a) Full trajectory vs baselines", loc="left", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
ax.grid(axis="y", alpha=0.2)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

for bar, a in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{a:.2f}", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold" if a > 0.9 else "normal")

# === (b) Cross-task transfer ===
ax = fig.add_subplot(gs[1])
tasks = ["TriviaQA", "HotpotQA", "GSM8K"]
ct_matrix = np.array([
    [0.789, 0.743, 0.380],
    [0.752, 0.789, 0.774],
    [0.559, 0.626, 0.789],
])
im = ax.imshow(ct_matrix, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(tasks, fontsize=9.5)
ax.set_yticklabels(tasks, fontsize=9.5)
ax.set_title("(b) Cross-task transfer", loc="left", fontweight="bold")
ax.set_xlabel("Test task", fontsize=10)
ax.set_ylabel("Train task", fontsize=10)

for i in range(3):
    for j in range(3):
        v = ct_matrix[i, j]
        color = "white" if v > 0.78 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color=color, fontsize=11, fontweight="bold")

# 【关键修复】Colorbar 标题放在顶部，不在侧面挤压右侧的 y 轴标签
cbar_b = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar_b.ax.set_title("AUC", fontsize=9, pad=6)

# === (c) Cross-model matrix ===
ax = fig.add_subplot(gs[2])
models = ["Qwen3-8B", "Llama-3.1-8B", "Mistral-7B"]
cm_matrix = np.array([
    [0.0, 0.728, 0.748],
    [0.695, 0.0, 0.779],
    [0.639, 0.599, 0.0],
])
display = cm_matrix.copy()
np.fill_diagonal(display, 0.95)
im2 = ax.imshow(display, cmap="Greens", vmin=0.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
ax.set_yticklabels(models, fontsize=9.5)
ax.set_title("(c) Cross-model transfer", loc="left", fontweight="bold")
ax.set_xlabel("Test model", fontsize=10)

for i in range(3):
    for j in range(3):
        if i == j:
            ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=13)
        else:
            v = cm_matrix[i, j]
            color = "white" if v > 0.78 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

# Colorbar 标题同样放在顶部
cbar_c = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cbar_c.ax.set_title("AUC", fontsize=9, pad=6)

plt.suptitle("Trajectory prefixes predict later signal decay",
             fontweight="bold", fontsize=12, y=0.97)

plt.savefig("./fig5_prediction.pdf", dpi=200)
plt.savefig("./fig5_prediction.png", dpi=200)
plt.close()
print("Figure 5 generated successfully.")
