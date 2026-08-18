"""
Full-depth (post-generation) feature audit on the 7-task server data.

Rationale: the non-oracle trigger flags errors AFTER the first generation
completes, so ALL layers' top-5 dynamics are available -- not just the prefix.
If gold ever competed, the eventual winner faced real competition, visible in
gold-free quantities: minimum top1-vs-top2 margin, second-half margin variance,
runner-up churn, and margin growth (winner's dominance increasing late).

Reports within-task held-out AUC: old (prefix-only) vs +fullDepth, per task.
Also reports the pooled AUC with GROUPED evaluation (task-stratified folds)
to check the task-identity inflation of the old pooled 0.669.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from run_inference_available_predictor import inference_available_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score


def full_depth_features(sample):
    top5 = sample.get("top5", [])
    if len(top5) < 6:
        return None
    n = len(top5)
    f = {}
    m12 = np.array([L[0][1] - L[1][1] for L in top5])
    f["m12_min_all"] = float(m12.min())
    f["m12_min_secondhalf"] = float(m12[n // 2:].min())
    f["m12_mean_secondhalf"] = float(m12[n // 2:].mean())
    f["m12_var_secondhalf"] = float(m12[n // 2:].var())
    r2 = [L[1][0] for L in top5]
    f["runnerup_churn_secondhalf"] = sum(
        1 for i in range(n // 2, n - 1) if r2[i] != r2[i + 1]) / max(n // 2 - 1, 1)
    t1 = [L[0][0] for L in top5]
    f["t1_churn_secondhalf"] = sum(
        1 for i in range(n // 2, n - 1) if t1[i] != t1[i + 1]) / max(n // 2 - 1, 1)
    f["m12_final"] = float(m12[-1])
    p5 = np.array([t[1] for t in top5[-1]]) + 1e-12
    p5 = p5 / p5.sum()
    f["entropy_final"] = float(-(p5 * np.log(p5)).sum())
    f["m12_growth"] = float(m12[-1] - m12.max())
    f["p2_at_min"] = float(top5[int(np.argmin(m12))][1][1])
    return f


FD_NAMES = ["m12_min_all", "m12_min_secondhalf", "m12_mean_secondhalf",
            "m12_var_secondhalf", "runnerup_churn_secondhalf",
            "t1_churn_secondhalf", "m12_final", "entropy_final",
            "m12_growth", "p2_at_min"]


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as fh:
        all_results = json.load(fh)

    combined, y, tasks = [], [], []
    for s in all_results:
        if s.get("final_correct", True):
            continue
        ranks = s.get("correct_rank", [])
        g = full_depth_features(s)
        b = inference_available_features(s)
        if g is None or b is None or len(ranks) < 2:
            continue
        combined.append({**g, **b})
        y.append(int(min(ranks[:-1]) <= config.RANK_COMPETITIVE))
        tasks.append(s.get("task", "?"))

    names = list(combined[0].keys())
    X = np.array([[f[k] for k in names] for f in combined])
    y = np.array(y)
    tasks = np.array(tasks)
    print(f"n={len(X)}, positives={y.sum()} ({100*y.mean():.1f}%)\n")

    def auc_within(Xe, ye):
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(500, max_depth=10, random_state=42, n_jobs=-1)
        return roc_auc_score(
            ye, cross_val_predict(rf, Xe, ye, cv=skf,
                                  method="predict_proba")[:, 1])

    old_idx = [i for i, n in enumerate(names) if n not in FD_NAMES]
    fd_idx = [i for i, n in enumerate(names) if n in FD_NAMES]

    print("[Within-task held-out AUC]")
    aucs_new = {}
    for t in np.unique(tasks):
        m = tasks == t
        if m.sum() < 40 or len(np.unique(y[m])) < 2:
            continue
        a_old = auc_within(X[np.ix_(m, old_idx)], y[m])
        a_fd = auc_within(X[np.ix_(m, fd_idx)], y[m])
        a_new = auc_within(X[m], y[m])
        aucs_new[t] = a_new
        print(f"  {t:16s} old={a_old:.3f}  fullDepth-only={a_fd:.3f}  "
              f"old+fullDepth={a_new:.3f}")
    print(f"  {'MEAN':16s} "
          f"old+fullDepth={np.mean(list(aucs_new.values())):.3f}")

    # Grouped evaluation: folds stratified so each fold contains all tasks
    # (removes the task-identity shortcut from the pooled number)
    print("\n[Pooled AUC, task-grouped folds (no task-identity shortcut)]")
    gkf = GroupKFold(5)
    for tag, idx in [("old (prefix-only)", old_idx), ("old+fullDepth", list(range(len(names))))]:
        rf = RandomForestClassifier(500, max_depth=10, random_state=42, n_jobs=-1)
        p = cross_val_predict(rf, X[:, idx], y, cv=gkf, groups=tasks,
                              method="predict_proba")[:, 1]
        print(f"  {tag:20s} grouped pooled AUC = {roc_auc_score(y, p):.3f}")

    rf = RandomForestClassifier(500, max_depth=10, random_state=42, n_jobs=-1).fit(X, y)
    imp = sorted(zip(names, rf.feature_importances_), key=lambda x: -x[1])[:8]
    print("\n[Top-8 feature importance]")
    for k, v in imp:
        print(f"  {k:26s} {float(v):.4f}")


if __name__ == "__main__":
    main()
