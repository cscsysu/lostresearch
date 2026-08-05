"""
Figure 3: Correct vs incorrect trajectories across three model families.
Shows rank and CIS trajectories for correct (blue) and incorrect (red) examples.
Generates: fig3_trajectories.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

DATA = {
    "qwen": ("Qwen3-8B", "full_results_Qwen3-8B.json"),
    "llama": ("Llama-3.1-8B", "cross_model_llama.json"),
    "mistral": ("Mistral-7B", "cross_model_mistral.json"),
}

def load(model_key):
    if model_key == "qwen":
        with open(f"/data/workspace/newiclr/lost-output/outputs/data/{DATA[model_key][1]}") as f:
            return json.load(f)
    else:
        with open(f"/data/workspace/newiclr/lost-output/outputs/data/{DATA[model_key][1]}") as f:
            d = json.load(f)
        return d["trajectory_results"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))

for col, (key, (name, _)) in enumerate(DATA.items()):
    data = load(key)
    correct = [s for s in data if s["final_correct"]]
    incorrect = [s for s in data if not s["final_correct"]]

    # === Top row: Rank trajectory (excl final layer) ===
    ax = axes[0, col]
    n_layers = max(len(s["correct_rank"]) for s in data)
    x = np.arange(n_layers)

    for group, color, label in [(correct, "#27AE60", "Correct"),
                                  (incorrect, "#C0392B", "Incorrect")]:
        # Compute median rank at each layer (using all examples that have it)
        medians = []
        for l in range(n_layers):
            ranks = [s["correct_rank"][l] for s in group if l < len(s["correct_rank"])]
            if ranks:
                medians.append(np.median(ranks))
            else:
                medians.append(np.nan)
        # Clip for log scale
        medians = [max(1, m) for m in medians]
        ax.plot(x, medians, color=color, label=label, linewidth=2, marker="o", markersize=3)

    ax.set_yscale("log")
    ax.set_title(f"{name}", fontweight="bold")
    ax.set_xlabel("Layer")
    if col == 0:
        ax.set_ylabel("Gold rank (log scale, lower=better)")
        ax.legend(loc="upper right", framealpha=0.9)
    # Add horizontal line at rank=5 (top-5 threshold)
    ax.axhline(5, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    if col == 2:
        ax.text(n_layers-1, 5.5, "top-5 threshold", fontsize=7, color="orange", ha="right", va="bottom")
    ax.grid(True, alpha=0.2)

    # === Bottom row: CIS trajectory ===
    ax = axes[1, col]
    for group, color, label in [(correct, "#27AE60", "Correct"),
                                  (incorrect, "#C0392B", "Incorrect")]:
        medians = []
        for l in range(n_layers):
            ciss = [s["cis"][l] for s in group if l < len(s["cis"])]
            if ciss:
                medians.append(np.median(ciss))
            else:
                medians.append(np.nan)
        ax.plot(x, medians, color=color, label=label, linewidth=2, marker="o", markersize=3)

    ax.axhline(0, color="black", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Layer")
    if col == 0:
        ax.set_ylabel("CIS (gold − competitor)")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("fig3_trajectories.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig3_trajectories.png", bbox_inches="tight", dpi=200)
plt.close()
print("Saved: figures/fig3_trajectories.pdf")
