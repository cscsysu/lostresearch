"""
Optimized non-oracle repairability predictor (v3).

Design history (all audited, per the result-reporting rule):
  v1: prefix-only top-5 stats. Pooled AUC 0.669, but a task-grouped audit
      showed much of it was task-identity leakage (within-task ~0.55).
  v2 attempts that did NOT help on local data: scale-free subset, shallow
      trees, few-shot calibration, generated-answer prefix trajectory.
  v3 (this file): use ALL layers (the trigger fires after the first
      generation, so the full trajectory is available -- there is no reason
      to restrict to the prefix) and add gold-free "competition dynamics"
      features that capture whether the eventual winner was challenged
      mid-network -- the observable signature of a preservation failure.

Every feature is gold-free and computable post-generation. We report:
  - within-task 5-fold held-out AUC (the honest deployment number),
  - task-grouped pooled AUC (removes the task-identity shortcut),
  - naive pooled AUC (for comparison with v1's 0.669),
  each with bootstrap 95% CIs, plus feature importance.

Usage:
  python run_nonoracle_predictor_v3.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


def competition_features(sample):
    """Gold-free, all-layer competition dynamics (post-generation)."""
    top5 = sample.get("top5", [])
    if len(top5) < 6:
        return None
    n = len(top5)
    f = {}
    win = top5[-1][0][0]  # eventual winner token id

    # --- winner's own trajectory across all layers ---
    win_rank, win_lp = [], []
    for L in top5:
        ids = [t[0] for t in L]
        if win in ids:
            j = ids.index(win)
            win_rank.append(j)
            win_lp.append(L[j][1])
        else:
            win_rank.append(5)
            win_lp.append(0.0)
    win_rank = np.array(win_rank, float)
    win_lp = np.array(win_lp, float)
    lead = np.where(win_rank == 0)[0]
    f["win_lead_layer"] = float(lead[0] / n) if len(lead) else 1.0
    f["win_notlead_frac"] = float(np.mean(win_rank > 0))
    f["win_lp_dipdepth"] = float(win_lp.max() - win_lp.min())
    f["win_lp_final_minus_mid"] = float(win_lp[-1] - win_lp[n // 2])

    # --- top1-vs-top2 margin dynamics (all layers) ---
    m12 = np.array([L[0][1] - L[1][1] for L in top5])
    f["m12_min"] = float(m12.min())
    f["m12_final"] = float(m12[-1])
    f["m12_dip_then_rise"] = float(m12[-1] - m12.min())
    f["m12_argmin_frac"] = float(np.argmin(m12) / n)
    f["m12_var2h"] = float(m12[n // 2:].var())
    f["m12_mean2h"] = float(m12[n // 2:].mean())

    # --- runner-up / leader churn ---
    r2 = [L[1][0] for L in top5]
    t1 = [L[0][0] for L in top5]
    f["r2_churn"] = sum(1 for i in range(n - 1) if r2[i] != r2[i + 1]) / (n - 1)
    f["t1_churn_2h"] = sum(1 for i in range(n // 2, n - 1)
                           if t1[i] != t1[i + 1]) / max(n // 2 - 1, 1)

    # --- entropy dynamics ---
    pe = []
    for L in top5:
        p = np.array([t[1] for t in L]) + 1e-12
        p = p / p.sum()
        pe.append(-(p * np.log(p)).sum())
    pe = np.array(pe)
    f["ent_min"] = float(pe.min())
    f["ent_final"] = float(pe[-1])
    f["ent_var2h"] = float(pe[n // 2:].var())
    f["ent_mean"] = float(pe.mean())
    return f


def boot_ci(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as fh:
        R = json.load(fh)

    X, y, tasks = [], [], []
    for s in R:
        if s.get("final_correct", True):
            continue
        rk = s.get("correct_rank", [])
        fe = competition_features(s)
        if fe is None or len(rk) < 2:
            continue
        X.append(fe)
        y.append(int(min(rk[:-1]) <= config.RANK_COMPETITIVE))
        tasks.append(s.get("task", "?"))
    names = list(X[0].keys())
    X = np.array([[f[k] for k in names] for f in X])
    y = np.array(y)
    tasks = np.array(tasks)
    print(f"n={len(X)}, positives={y.sum()} ({100*y.mean():.1f}%), "
          f"tasks={len(np.unique(tasks))}\n")

    def rf():
        return RandomForestClassifier(500, max_depth=8, random_state=42, n_jobs=-1)

    # --- within-task ---
    print("[Within-task 5-fold held-out AUC] (honest deployment number)")
    within = {}
    for t in np.unique(tasks):
        m = tasks == t
        if m.sum() < 40 or len(np.unique(y[m])) < 2:
            continue
        p = cross_val_predict(rf(), X[m], y[m],
                              cv=StratifiedKFold(5, shuffle=True, random_state=42),
                              method="predict_proba")[:, 1]
        a = roc_auc_score(y[m], p)
        lo, hi = boot_ci(y[m], p)
        within[t] = a
        print(f"  {t:16s} AUC={a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={m.sum()})")
    if within:
        print(f"  {'MEAN':16s} AUC={np.mean(list(within.values())):.3f}")

    # --- task-grouped pooled ---
    n_groups = len(np.unique(tasks))
    if n_groups >= 2:
        gkf = GroupKFold(min(5, n_groups))
        p = cross_val_predict(rf(), X, y, cv=gkf, groups=tasks,
                              method="predict_proba")[:, 1]
        a = roc_auc_score(y, p)
        lo, hi = boot_ci(y, p)
        print(f"\n[Task-grouped pooled AUC] (no task-identity shortcut) "
              f"= {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    # --- naive pooled (v1-style) ---
    p = cross_val_predict(rf(), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          method="predict_proba")[:, 1]
    a = roc_auc_score(y, p)
    lo, hi = boot_ci(y, p)
    print(f"[Naive pooled AUC] (v1-style, task-mixed) "
          f"= {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    clf = rf().fit(X, y)
    imp = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])[:10]
    print("\n[Top-10 feature importance]")
    for k, v in imp:
        print(f"  {k:24s} {float(v):.4f}")

    out = os.path.join(config.DATA_DIR, "nonoracle_predictor_v3_Qwen3-8B.json")
    with open(out, "w") as fh:
        json.dump({"within_task": within,
                   "feature_importance": {k: float(v) for k, v in imp}}, fh, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
