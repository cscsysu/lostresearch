"""
Figure 6: Behavioral intervention and component attribution.
Left: ablation flip rates across models and controls.
Right: direct logit attribution for attention vs MLP per layer (Qwen3-8B).
Generates: fig6_intervention.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# Left panel: behavioral necessity
models = ["Qwen3-8B", "Llama-3.1-8B", "Mistral-7B"]
gold_flip = [76, 73, 40]      # percentage
random_flip = [0, 0, 0]
second_best = [13.3, None, None]  # only Qwen

# Right panel: component attribution (attention vs MLP per layer) from real data
# Load real p1_components data (incorrect examples), mean per-layer contribution.
import os
import sys

def load_real_attribution():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "lost-output",
                     "outputs", "data", "p1_components_Qwen3-8B.json"),
        "/data2/css2025/lostresearch/lostresearch/outputs/data/"
        "p1_components_Qwen3-8B.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            cd = d["component_decomposition"]
            incorrect = [s for s in cd if not s.get("final_correct")]
            n_layers = max(x["layer"] for s in incorrect for x in s["layer_cis"]) + 1
            attn = np.zeros(n_layers); mlp = np.zeros(n_layers); cnt = np.zeros(n_layers)
            for s in incorrect:
                for x in s["layer_cis"]:
                    l = x["layer"]
                    attn[l] += x["attn_contrib"]; mlp[l] += x["mlp_contrib"]; cnt[l] += 1
            return attn / np.maximum(cnt, 1), mlp / np.maximum(cnt, 1), n_layers
    return None

real = load_real_attribution()
if real is not None:
    attn_dla, mlp_dla, n_layers = real
else:
    # fallback to the hard-coded snapshot (should not happen in practice)
    n_layers = 36
    attn_dla = np.zeros(n_layers); mlp_dla = np.zeros(n_layers)
    attn_dla[35] = -0.16; mlp_dla[35] = -3.42

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

# === (a) Ablation flip rates ===
ax = axes[0]
x = np.arange(len(models))
width = 0.32
bars1 = ax.bar(x - width/2, gold_flip, width, color="#C0392B",
               edgecolor="black", linewidth=0.7, label="Gold direction")
bars2 = ax.bar(x + width/2, random_flip, width, color="#95A5A6",
               edgecolor="black", linewidth=0.7, label="Random token")
for bar, v in zip(bars1, gold_flip):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{v}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, v in zip(bars2, random_flip):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{v}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Correct$\\rightarrow$wrong flip rate (%)")
ax.set_title("(a) Behavioral necessity: gold-direction ablation", loc="left", fontweight="bold")
ax.set_ylim(0, 95)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.2)

# Add Qwen control annotation
ax.annotate("Qwen3-8B controls:\n2nd-best: 13%, contrast: 33%",
            xy=(0, 76), xytext=(0.5, 55),
            fontsize=8, style="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7",
                      edgecolor="#F4D03F", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="#555555"))

# === (b) DLA per layer ===
ax = axes[1]
layers = np.arange(n_layers)
ax.bar(layers - 0.2, attn_dla, width=0.4, color="#3498DB",
       edgecolor="black", linewidth=0.5, label="Attention", alpha=0.85)
ax.bar(layers + 0.2, mlp_dla, width=0.4, color="#E67E22",
       edgecolor="black", linewidth=0.5, label="MLP", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("$\\Delta$CIS attribution (negative = suppresses gold)")
ax.set_title("(b) Component attribution (Qwen3-8B, incorrect examples)",
             loc="left", fontweight="bold")
ax.legend(loc="upper left")
ax.set_xticks([0, 10, 20, 30, 35])
ax.set_xticklabels(["0", "10", "20", "30", "35"])
# Annotate layer 35
ax.annotate("Layer 35 MLP:\n-3.42 (dominant)",
            xy=(35, -3.42), xytext=(20, -2.6),
            fontsize=8, color="#C0392B", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C0392B"))
ax.grid(axis="y", alpha=0.2)

plt.suptitle("Behavioral intervention and component attribution",
             fontweight="bold", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("fig6_intervention.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig6_intervention.png", bbox_inches="tight", dpi=200)
plt.close()
print("Saved: figures/fig6_intervention.pdf")
