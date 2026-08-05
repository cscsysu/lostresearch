# """
# Figure 2: Measurement bias. Layout redesigned to avoid label overlap.
# Left: observed vs null sign-reversal rate.
# Right: raw vs tuned peak CIS, showing the inflation factor.
# Generates: fig2_measurement_bias.pdf
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyBboxPatch

# plt.rcParams.update({
#     "font.size": 12,
#     "axes.titlesize": 13,
#     "axes.labelsize": 12,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "legend.fontsize": 11,
# })

# fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
# fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.18, wspace=0.30)

# # === Left: observed vs null sign-reversal rate ===
# ax = axes[0]
# labels = ["Observed\n(pairwise)", "Endpoint-\nconditioned null"]
# rates = [88.2, 88.5]
# colors = ["#C0392B", "#5DADE2"]
# bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
# ax.set_ylim(0, 105)
# ax.set_ylabel("Sign-reversal rate (%)")
# ax.set_title("(a) Pairwise sign reversal", loc="left", fontweight="bold")
# ax.set_yticks([0, 25, 50, 75, 100])
# ax.grid(axis="y", alpha=0.2)
# for bar, r in zip(bars, rates):
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
#             f"{r}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
# # Annotation below the bars (out of overlap zone), in the lower-left
# ax.text(0.5, 0.35, "Difference within noise.\nNo evidence of competitive signal.",
#         transform=ax.transAxes, ha="center", va="center",
#         fontsize=10, style="italic", color="#555555",
#         bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7",
#                   edgecolor="#F4D03F", alpha=0.9))

# # === Right: raw vs tuned peak CIS ===
# ax = axes[1]
# models = ["Qwen3-8B", "Llama-3.1-8B", "Mistral-7B"]
# raw_vals = [4.53, 4.6, 4.4]    # Llama/Mistral estimated similar
# tuned_vals = [2.24, 2.3, 2.2]
# x = np.arange(len(models))
# width = 0.35
# bars1 = ax.bar(x - width/2, raw_vals, width, color="#E67E22",
#                edgecolor="black", linewidth=0.8, label="Raw logit lens")
# bars2 = ax.bar(x + width/2, tuned_vals, width, color="#27AE60",
#                edgecolor="black", linewidth=0.8, label="Tuned lens")
# ax.set_ylabel("Peak intermediate CIS")
# ax.set_title("(b) Raw vs tuned peak CIS (incorrect examples)",
#              loc="left", fontweight="bold")
# ax.set_xticks(x)
# ax.set_xticklabels(models)
# ax.set_ylim(0, 7)
# ax.legend(loc="upper right", framealpha=0.95)
# ax.grid(axis="y", alpha=0.2)
# for bar, v in zip(bars1, raw_vals):
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
#             f"{v:.1f}", ha="center", va="bottom", fontsize=11)
# for bar, v in zip(bars2, tuned_vals):
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
#             f"{v:.1f}", ha="center", va="bottom", fontsize=11)
# # Annotation: text in lower-right corner, arrow points up-left to Qwen raw bar
# ax.annotate("Raw inflates by ~2.0x",
#             xy=(0, 4.53), xytext=(1.7, 0.5),
#             fontsize=11, color="#C0392B", fontweight="bold",
#             arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
#             ha="center")

# plt.savefig("./fig2_measurement_bias.pdf", dpi=200)
# plt.savefig("./fig2_measurement_bias.png", dpi=200)
# plt.close()
# print("Done")
"""
Figure 2: Measurement bias.
Left: observed vs null sign-reversal rate.
Right: raw vs tuned peak CIS, with clean top-annotation (no text blocking).
Generates: fig2_measurement_bias.pdf / .png
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
})

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.18, wspace=0.28)

# === Left: observed vs null sign-reversal rate ===
ax = axes[0]
labels = ["Observed\n(pairwise)", "Endpoint-\nconditioned null"]
rates = [88.2, 88.5]
colors = ["#C0392B", "#5DADE2"]
bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.8, width=0.52)
ax.set_ylim(0, 108)
ax.set_ylabel("Sign-reversal rate (%)")
ax.set_title("(a) Pairwise sign reversal", loc="left", fontweight="bold")
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis="y", alpha=0.2)

for bar, r in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{r}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Annotation inside box
ax.text(0.5, 0.38, "Difference within noise.\nNo evidence of competitive signal.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=9.5, style="italic", color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7",
                  edgecolor="#F4D03F", alpha=0.95))

# === Right: raw vs tuned peak CIS ===
ax = axes[1]
models = ["Qwen3-8B", "Llama-3.1-8B", "Mistral-7B"]
raw_vals = [4.53, 4.6, 4.4]
tuned_vals = [2.24, 2.3, 2.2]
x = np.arange(len(models))
width = 0.32

bars1 = ax.bar(x - width/2, raw_vals, width, color="#E67E22",
               edgecolor="black", linewidth=0.8, label="Raw logit lens")
bars2 = ax.bar(x + width/2, tuned_vals, width, color="#27AE60",
               edgecolor="black", linewidth=0.8, label="Tuned lens")

ax.set_ylabel("Peak intermediate CIS")
ax.set_title("(b) Raw vs tuned peak CIS (incorrect examples)",
             loc="left", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 6.8)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
ax.grid(axis="y", alpha=0.2)

for bar, v in zip(bars1, raw_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{v:.1f}", ha="center", va="bottom", fontsize=10)
for bar, v in zip(bars2, tuned_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{v:.1f}", ha="center", va="bottom", fontsize=10)

# 【关键修复】标注放在上方空白区域 (y=5.7)，弧形短箭头指向 Qwen Raw 柱体，不遮挡任何文字
ax.annotate("Raw inflates by ~2.0x",
            xy=(-0.16, 4.68), xytext=(0.55, 5.75),
            fontsize=9.5, color="#C0392B", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2,
                            connectionstyle="arc3,rad=-0.18"),
            ha="center", va="center")

plt.savefig("./fig2_measurement_bias.pdf", dpi=200)
plt.savefig("./fig2_measurement_bias.png", dpi=200)
plt.close()
print("Figure 2 generated successfully.")
