"""
Inference-available (non-oracle) repairability predictor. (v2, leak-audited)

KEY CONTRIBUTION: A trigger that requires NO gold answer at inference time.

v2 changes (reviewer round 6):
  - AUDITED every feature for future-information leakage. All features are
    computed strictly from the FIRST-HALF layers' top-k distributions; the
    anchor for leader-hold is the top-1 at the LAST PREFIX layer (mid-network),
    never the final-layer output. Renamed to `prefix_leader_hold` to make the
    prefix-only semantics explicit (the earlier name/paper wording suggested
    an "eventual leader" anchor, which would have leaked future information;
    the code never used it, but the naming is now unambiguous).
  - Added bootstrap 95% CI for the AUC.
  - Added cross-task generalization (train on one task, test on another).
  - Saves out-of-fold repair probabilities per sample so a downstream
    intervention script can use this predictor as a real trigger.

Features (all prefix-only, no gold token, no final-layer identity):
  - top-1 identity switching rate across prefix layers
  - prefix-leader hold fraction (anchor = last prefix layer's top-1)
  - top-1 probability trajectory (peak, slope, variance)
  - top-1 vs top-2 margin dynamics
  - top-5 candidate-set churn
  - entropy of top-5 distribution
  - total top-5 probability mass

Usage:
  python run_inference_available_predictor.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score


def inference_available_features(sample, t0_frac=0.5):
    """Extract features from FIRST-HALF per-layer top-k distributions ONLY.

    Leakage audit: nothing here references the gold token, the final-layer
    competitor, the final layer itself, or any layer beyond index t0-1.
    The 'prefix leader' anchor is the top-1 at the last prefix layer.
    """
    top5 = sample.get("top5", [])
    n = len(top5)
    if n < 6:
        return None
    t0 = max(3, int(n * t0_frac))
    pre = top5[:t0]

    f = {}
    t1_ids = [L[0][0] for L in pre]
    t1_ps = [L[0][1] for L in pre]
    m12 = [L[0][1] - L[1][1] for L in pre if len(L) >= 2]

    # --- Leader (top-1) identity dynamics, prefix only ---
    f["switch_rate"] = sum(1 for i in range(1, len(t1_ids))
                           if t1_ids[i] != t1_ids[i - 1]) / max(len(t1_ids) - 1, 1)
    f["n_distinct_leaders"] = len(set(t1_ids))
    # anchor = top-1 at the LAST PREFIX layer (mid-network), not final layer
    prefix_leader = t1_ids[-1]
    f["prefix_leader_hold"] = sum(1 for tid in t1_ids
                                  if tid == prefix_leader) / len(t1_ids)

    # --- Leader confidence dynamics (prefix only) ---
    f["t1_prob_end"] = t1_ps[-1]
    f["t1_prob_max"] = max(t1_ps)
    f["t1_prob_mean"] = float(np.mean(t1_ps))
    f["t1_prob_var"] = float(np.var(t1_ps))
    f["t1_prob_slope"] = (t1_ps[-1] - t1_ps[0]) / max(len(t1_ps) - 1, 1)
    f["t1_peak_to_end"] = max(t1_ps) - t1_ps[-1]

    # --- Competition intensity (top-1 vs top-2, prefix only) ---
    f["margin12_end"] = m12[-1]
    f["margin12_min"] = min(m12)
    f["margin12_mean"] = float(np.mean(m12))
    f["margin12_var"] = float(np.var(m12))
    f["close_race_frac"] = sum(1 for m in m12 if m < 0.1) / len(m12)

    # --- Candidate-set churn (prefix only) ---
    churn = []
    for i in range(1, len(pre)):
        a = set(t[0] for t in pre[i - 1])
        b = set(t[0] for t in pre[i])
        churn.append(1 - len(a & b) / 5.0)
    f["churn_mean"] = float(np.mean(churn))
    f["churn_end"] = churn[-1]
    f["churn_max"] = max(churn)

    # --- Distributional uncertainty (prefix only) ---
    ents = []
    for L in pre:
        p = np.array([t[1] for t in L]) + 1e-12
        p = p / p.sum()
        ents.append(float(-(p * np.log(p)).sum()))
    f["entropy_end"] = ents[-1]
    f["entropy_mean"] = float(np.mean(ents))
    f["entropy_slope"] = (ents[-1] - ents[0]) / max(len(ents) - 1, 1)

    # --- Probability mass concentration (prefix only) ---
    mass = [sum(t[1] for t in L) for L in pre]
    f["mass_end"] = mass[-1]
    f["mass_mean"] = float(np.mean(mass))
    return f


def load_xy(results_file=None):
    if results_file is None:
        results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
        if not os.path.exists(results_file):
            alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                               "outputs", "data", "full_results_Qwen3-8B.json")
            results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as fh:
        all_results = json.load(fh)

    errors = [s for s in all_results if not s.get("final_correct")]
    print(f"Total errors: {len(errors)}")

    X_list, y_list, tasks, ids = [], [], [], []
    for s in errors:
        fe = inference_available_features(s)
        ranks = s.get("correct_rank", [])
        if fe is None or len(ranks) < 2:
            continue
        is_pres = min(ranks[:-1]) <= config.RANK_COMPETITIVE
        X_list.append(fe)
        y_list.append(int(is_pres))
        tasks.append(s.get("task", "unknown"))
        ids.append(s["id"])

    names = list(X_list[0].keys())
    X = np.array([[fe[k] for k in names] for fe in X_list])
    y = np.array(y_list)
    return X, y, np.array(tasks), ids, names


def bootstrap_auc_ci(y, proba, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    X, y, tasks, ids, names = load_xy()
    print(f"Samples: {len(X)}, features: {len(names)}")
    print(f"Preservation (repairable): {y.sum()} ({100*y.mean():.1f}%)")

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    print("\n" + "=" * 70)
    print("NON-ORACLE REPAIRABILITY PREDICTOR (prefix-only, leak-audited)")
    print("=" * 70)

    best_name, best_auc, best_clf = None, 0.0, None
    for nm, clf, need_scale in [
        ("Logistic (L2)", LogisticRegression(max_iter=5000), True),
        ("RandomForest", RandomForestClassifier(500, max_depth=10, random_state=42), False),
        ("GradientBoost", GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                     random_state=42), False),
    ]:
        Xu = StandardScaler().fit_transform(X) if need_scale else X
        from sklearn.model_selection import cross_val_score
        sc = cross_val_score(clf, Xu, y, cv=skf, scoring="roc_auc")
        print(f"  {nm:16s} ROC-AUC = {sc.mean():.3f} +/- {sc.std():.3f}")
        if sc.mean() > best_auc:
            best_name, best_auc, best_clf = nm, sc.mean(), (clf, need_scale)

    clf, need_scale = best_clf
    Xu = StandardScaler().fit_transform(X) if need_scale else X
    proba = cross_val_predict(clf, Xu, y, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    pr = average_precision_score(y, proba)
    lo, hi = bootstrap_auc_ci(y, proba)
    print(f"\n  Best: {best_name}, pooled out-of-fold ROC-AUC {auc:.3f} "
          f"(bootstrap 95% CI [{lo:.3f}, {hi:.3f}]), PR-AUC {pr:.3f}")
    print(f"  (baseline PR-AUC = positive rate = {y.mean():.3f})")

    # --- Cross-task generalization (leave-one-task-out) ---
    print("\n  Cross-task generalization (train on one task, test on another):")
    xt_results = {}
    unique_tasks = sorted(set(tasks))
    for tr in unique_tasks:
        for te in unique_tasks:
            if tr == te:
                continue
            m_tr, m_te = tasks == tr, tasks == te
            if m_tr.sum() < 30 or m_te.sum() < 30 or len(np.unique(y[m_tr])) < 2 \
               or len(np.unique(y[m_te])) < 2:
                continue
            c = RandomForestClassifier(500, max_depth=10, random_state=42)
            c.fit(X[m_tr], y[m_tr])
            a = roc_auc_score(y[m_te], c.predict_proba(X[m_te])[:, 1])
            xt_results[f"{tr}->{te}"] = float(a)
            print(f"    {tr:14s} -> {te:14s} AUC = {a:.3f}")

    # --- Selective-intervention utility ---
    print("\n  Selective-intervention utility (precision at coverage):")
    order = np.argsort(-proba)
    for cov in [0.1, 0.2, 0.3, 0.5]:
        k = int(len(order) * cov)
        sel = order[:k]
        prec = y[sel].mean()
        print(f"    Top {int(cov*100):2d}% flagged: precision = {prec:.3f} "
              f"(lift = {prec/y.mean():.2f}x over random)")

    # --- Feature importance ---
    rf = RandomForestClassifier(500, max_depth=10, random_state=42).fit(X, y)
    imp = sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1])[:10]
    print("\n  Top-10 features:")
    for k, v in imp:
        print(f"    {k:22s} {v:.4f}")

    out_file = os.path.join(config.DATA_DIR, "inference_available_predictor_Qwen3-8B.json")
    with open(out_file, "w") as fh:
        json.dump({
            "n": len(X), "n_features": len(names),
            "positive_rate": float(y.mean()),
            "best_model": best_name,
            "roc_auc_pooled": float(auc),
            "roc_auc_ci": [lo, hi],
            "pr_auc": float(pr),
            "cross_task": xt_results,
            "feature_importance": {k: float(v) for k, v in imp},
            "oof": [{"id": i, "task": t, "p": float(p), "y": int(yy)}
                    for i, t, p, yy in zip(ids, tasks, proba, y)],
        }, fh, indent=2)
    print(f"\nSaved (incl. out-of-fold probabilities for intervention driving): {out_file}")


if __name__ == "__main__":
    main()
