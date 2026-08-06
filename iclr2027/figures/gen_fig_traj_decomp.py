"""
Trajectory decomposition: why does the trajectory beat any snapshot?

Answers reviewer's core question: trajectory prediction is not a black-box
feature, we can decompose it. We show the decay-prediction AUC of single
trajectory features (peak, slope, variance, oscillation, rank, early
confidence) and combinations, up to the full trajectory, revealing which
dynamic aspects of the trajectory carry the predictive signal.

Generates: fig_trajectory_decomposition.pdf (AUC bar chart)
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "lostresearch"))
import config
from prediction import extract_features, extract_targets


def load_and_compute(data_path):
    with open(data_path) as f:
        d = json.load(f)
    samples = d if isinstance(d, list) else d.get("trajectory_results", [])
    feats, y = [], []
    for s in samples:
        f = extract_features(s, config.PREDICTION_T0)
        t = extract_targets(s)
        if f and t:
            feats.append(f)
            y.append(t["will_decay"])
    y = np.array(y)
    names = list(feats[0].keys())
    X = {k: np.array([x[k] for x in feats]) for k in names}
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    def cv(Xmat):
        sc = StandardScaler()
        Xs = sc.fit_transform(Xmat)
        return cross_val_score(LogisticRegression(max_iter=2000), Xs, y,
                               cv=skf, scoring="roc_auc").mean()

    # Single features (direction-agnostic).
    single = {}
    for k in names:
        X1 = X[k].reshape(-1, 1)
        single[k] = max(cv(X1), cv(-X1))

    # Combinations.
    combos = {
        "Peak": ["cis_max_before"],
        "Slope": ["cis_slope"],
        "Variance": ["cis_variance"],
        "Oscillation": ["transitions"],
        "Early conf.": ["cis_at_t0"],
        "Peak+Slope": ["cis_max_before", "cis_slope"],
        "Slope+Var": ["cis_slope", "cis_variance"],
        "Peak+Slope+Var": ["cis_max_before", "cis_slope", "cis_variance"],
        "Full trajectory": names,
    }
    combo_auc = {}
    for name, cols in combos.items():
        Xm = np.stack([X[c] for c in cols], axis=1)
        combo_auc[name] = cv(Xm)

    return single, combo_auc


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "..", "..", "lost-output",
                     "outputs", "data", "full_results_Qwen3-8B.json")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    single, combo = load_and_compute(data_path)

    # Build the bar chart: progressive feature inclusion up to full trajectory.
    order = ["Peak", "Slope", "Variance", "Oscillation", "Early conf.",
             "Peak+Slope", "Slope+Var", "Peak+Slope+Var", "Full trajectory"]
    aucs = [combo[o] for o in order]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                         "axes.labelsize": 10, "xtick.labelsize": 9})
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#95A5A6"] * 8 + ["#C0392B"]
    bars = ax.bar(order, aucs, color=colors, edgecolor="black", linewidth=0.7)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.4, 0.82)
    ax.set_ylabel("Decay-prediction AUC")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("Trajectory decomposition: dynamic shape, not a snapshot",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_traj_decomp.pdf"),
                bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(out_dir, "fig_traj_decomp.png"),
                bbox_inches="tight", dpi=200)
    print("Saved fig_traj_decomp")
    print("\nSingle-feature AUCs:")
    for k, v in sorted(single.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:.3f}")


if __name__ == "__main__":
    main()
