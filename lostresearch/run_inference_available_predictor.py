"""
Inference-available (non-oracle) repairability predictor.

KEY CONTRIBUTION: A trigger that requires NO gold answer at inference time.

Previous predictors used gold-token features (log p(y*), rank(y*), CIS),
which are unavailable at deployment. This predictor uses ONLY the per-layer
top-k output distribution -- information the model produces anyway.

Target: "Is this error repairable?" (preservation vs formation),
which is the decision a practitioner actually needs to make.

Features (all inference-available):
  - top-1 identity switching rate across layers
  - top-1 probability trajectory (peak, slope, variance)
  - top-1 vs top-2 margin dynamics (competition intensity)
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
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import roc_auc_score, average_precision_score


def inference_available_features(sample, t0_frac=0.5):
    """Extract features from per-layer top-k distributions ONLY.

    No gold token, no final-layer competitor identity, no correctness label.
    Everything here is computable during a normal forward pass.
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

    # --- Leader (top-1) identity dynamics ---
    f["switch_rate"] = sum(1 for i in range(1, len(t1_ids))
                           if t1_ids[i] != t1_ids[i - 1]) / max(len(t1_ids) - 1, 1)
    f["n_distinct_leaders"] = len(set(t1_ids))
    last = t1_ids[-1]
    hold = 0
    for tid in reversed(t1_ids):
        if tid == last:
            hold += 1
        else:
            break
    f["leader_hold_frac"] = hold / len(t1_ids)

    # --- Leader confidence dynamics ---
    f["t1_prob_final"] = t1_ps[-1]
    f["t1_prob_max"] = max(t1_ps)
    f["t1_prob_mean"] = float(np.mean(t1_ps))
    f["t1_prob_var"] = float(np.var(t1_ps))
    f["t1_prob_slope"] = (t1_ps[-1] - t1_ps[0]) / max(len(t1_ps) - 1, 1)
    f["t1_peak_to_final"] = max(t1_ps) - t1_ps[-1]

    # --- Competition intensity (top-1 vs top-2) ---
    f["margin12_final"] = m12[-1]
    f["margin12_min"] = min(m12)
    f["margin12_mean"] = float(np.mean(m12))
    f["margin12_var"] = float(np.var(m12))
    f["close_race_frac"] = sum(1 for m in m12 if m < 0.1) / len(m12)

    # --- Candidate-set churn ---
    churn = []
    for i in range(1, len(pre)):
        a = set(t[0] for t in pre[i - 1])
        b = set(t[0] for t in pre[i])
        churn.append(1 - len(a & b) / 5.0)
    f["churn_mean"] = float(np.mean(churn))
    f["churn_final"] = churn[-1]
    f["churn_max"] = max(churn)

    # --- Distributional uncertainty ---
    ents = []
    for L in pre:
        p = np.array([t[1] for t in L]) + 1e-12
        p = p / p.sum()
        ents.append(float(-(p * np.log(p)).sum()))
    f["entropy_final"] = ents[-1]
    f["entropy_mean"] = float(np.mean(ents))
    f["entropy_slope"] = (ents[-1] - ents[0]) / max(len(ents) - 1, 1)

    # --- Probability mass concentration ---
    mass = [sum(t[1] for t in L) for L in pre]
    f["mass_final"] = mass[-1]
    f["mass_mean"] = float(np.mean(mass))
    return f


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as fh:
        all_results = json.load(fh)

    errors = [s for s in all_results if not s.get("final_correct")]
    print(f"Total errors: {len(errors)}")

    X_list, y_list, tasks = [], [], []
    for s in errors:
        fe = inference_available_features(s)
        ranks = s.get("correct_rank", [])
        if fe is None or len(ranks) < 2:
            continue
        is_pres = min(ranks[:-1]) <= config.RANK_COMPETITIVE
        X_list.append(fe)
        y_list.append(int(is_pres))
        tasks.append(s.get("task", "unknown"))

    names = list(X_list[0].keys())
    X = np.array([[fe[k] for k in names] for fe in X_list])
    y = np.array(y_list)
    print(f"Samples: {len(X)}, features: {len(names)}")
    print(f"Preservation (repairable): {y.sum()} ({100*y.mean():.1f}%)")

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    print("\n" + "=" * 70)
    print("NON-ORACLE REPAIRABILITY PREDICTOR")
    print("(uses only per-layer top-k distributions; no gold token)")
    print("=" * 70)

    best_name, best_auc, best_clf = None, 0.0, None
    for nm, clf, need_scale in [
        ("Logistic (L2)", LogisticRegression(max_iter=5000), True),
        ("RandomForest", RandomForestClassifier(500, max_depth=10, random_state=42), False),
        ("GradientBoost", GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                     random_state=42), False),
    ]:
        Xu = StandardScaler().fit_transform(X) if need_scale else X
        sc = cross_val_score(clf, Xu, y, cv=skf, scoring="roc_auc")
        print(f"  {nm:16s} ROC-AUC = {sc.mean():.3f} +/- {sc.std():.3f}")
        if sc.mean() > best_auc:
            best_name, best_auc, best_clf = nm, sc.mean(), (clf, need_scale)

    # PR-AUC for the best model
    clf, need_scale = best_clf
    Xu = StandardScaler().fit_transform(X) if need_scale else X
    proba = cross_val_predict(clf, Xu, y, cv=skf, method="predict_proba")[:, 1]
    pr = average_precision_score(y, proba)
    print(f"\n  Best: {best_name}, ROC-AUC {best_auc:.3f}, PR-AUC {pr:.3f}")
    print(f"  (baseline PR-AUC = positive rate = {y.mean():.3f})")

    # Precision at different coverage levels (practical utility curve)
    print("\n  Selective-intervention utility (precision at coverage):")
    order = np.argsort(-proba)
    for cov in [0.1, 0.2, 0.3, 0.5]:
        k = int(len(order) * cov)
        sel = order[:k]
        prec = y[sel].mean()
        print(f"    Top {int(cov*100):2d}% flagged: precision = {prec:.3f} "
              f"(lift = {prec/y.mean():.2f}x over random)")

    # Feature importance
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
            "best_model": best_name, "roc_auc": float(best_auc),
            "pr_auc": float(pr),
            "feature_importance": {k: float(v) for k, v in imp},
        }, fh, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
