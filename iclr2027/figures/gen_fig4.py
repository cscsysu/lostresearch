"""
Figure 4: Failure taxonomy by task — formation vs preservation vs successful.
Shows the breakdown for each task in the Qwen analysis.
Generates: fig4_taxonomy.pdf
"""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("/data/workspace/newiclr/lost-output/outputs/data/full_results_Qwen3-8B.json") as f:
    data = json.load(f)

incorrect = [s for s in data if not s["final_correct"]]
correct = [s for s in data if s["final_correct"]]

def classify(s):
    """formation / preservation / successful."""
    cis = s["cis"]
    ranks = s["correct_rank"]
    if len(cis) < 4:
        return "other"
    mid_cis = cis[1:-1]
    mid_ranks = ranks[1:-1]
    has_competitive = any(
        mid_cis[i] > 0 and mid_ranks[i] <= 4
        for i in range(len(mid_cis))
    )
    final_neg = cis[-1] < 0
    if has_competitive and final_neg:
        return "preservation"
    elif not has_competitive:
        return "formation"
    elif has_competitive and not final_neg:
        return "successful"
    return "other"

# Classify correct and incorrect
for s in data:
    s["_class"] = classify(s)

tasks = ["triviaqa", "hotpotqa", "gsm8k"]
categories = ["formation", "preservation", "successful"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

task_names = {"triviaqa": "TriviaQA\n(knowledge QA)",
              "hotpotqa": "HotpotQA\n(multi-hop QA)",
              "gsm8k": "GSM8K\n(math reasoning)"}

# 收集每任务数据
for col, task in enumerate(tasks):
    ax = axes[col]
    # 统计每类
    inc_task = [s for s in incorrect if s.get("task") == task]
    cor_task = [s for s in correct if s.get("task") == task]
    n_total = len(inc_task) + len(cor_task)

    # Incorrect: formation / preservation
    n_formation = sum(1 for s in inc_task if s["_class"] == "formation")
    n_preservation = sum(1 for s in inc_task if s["_class"] == "preservation")
    # Correct: successful
    n_successful = len(cor_task)

    # Stacked bar
    labels = ["Formation\nfailure", "Preservation\nfailure", "Successful\npreservation"]
    values = [n_formation, n_preservation, n_successful]
    colors = ["#C0392B", "#E67E22", "#27AE60"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, values):
        pct = 100 * v / n_total if n_total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + n_total*0.01,
                f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(task_names[task], fontweight="bold")
    ax.set_ylabel("# examples")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.2)

plt.suptitle("Failure taxonomy by task (Qwen3-8B)", fontweight="bold", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("fig4_taxonomy.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig4_taxonomy.png", bbox_inches="tight", dpi=200)
plt.close()
print("Saved: figures/fig4_taxonomy.pdf")
