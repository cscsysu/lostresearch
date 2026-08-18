"""
Cross-task transfer audit for the inference-available predictor.

Hypotheses tested (per user's rule: audit before writing negative results):
  H1: absolute-scale features fail to transfer; scale-free features transfer
      better. Compare full feature set vs scale-free subset.
  H2: without a within-task reference point, "transfer ~= chance" is
      uninterpretable. Measure within-task held-out AUC per task.
  H3: GSM8K directions (12 positives) are statistical noise; report with and
      without them.
  H4: few-shot target-task calibration (deploy-standard) rescues transfer:
      train on source + 50 labeled target samples, test on the rest.

Runs locally from stored top5 trajectories; no GPU needed.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from run_inference_available_predictor import inference_available_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


SCALE_FREE = ["switch_rate", "n_distinct_leaders", "prefix_leader_hold",
              "t1_prob_slope", "t1_peak_to_end", "close_race_frac",
              "churn_mean", "churn_end", "entropy_slope", "mass_mean"]


def load():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as fh:
        all_results = json.load(fh)

    X, y, tasks = [], [], []
    for s in all_results:
        if s.get("final_correct", True):
            continue
        fe = inference_available_features(s)
        ranks = s.get("correct_rank", [])
        if fe is None or len(ranks) < 2:
            continue
        X.append(fe)
        y.append(int(min(ranks[:-1]) <= config.RANK_COMPETITIVE))
        tasks.append(s.get("task", "?"))
    names = list(X[0].keys())
    X = np.array([[f[k] for k in names] for f in X])
    return X, np.array(y), np.array(tasks), names


def rf(depth=10):
    return RandomForestClassifier(500, max_depth=depth, random_state=42,
                                  n_jobs=-1)


def main():
    X, y, tasks, names = load()
    print(f"n={len(X)}, positives={y.sum()} ({100*y.mean():.1f}%)")
    for t in np.unique(tasks):
        m = tasks == t
        print(f"  {t:14s} n={m.sum():4d} pos={y[m].sum():3d} "
              f"({100*y[m].mean():.1f}%)")

    # ---- H2: within-task held-out AUC (reference point) ----
    print("\n[H2] Within-task 5-fold held-out AUC (the transfer reference):")
    within = {}
    for t in np.unique(tasks):
        m = tasks == t
        if m.sum() < 40 or len(np.unique(y[m])) < 2:
            continue
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        p = cross_val_predict(rf(), X[m], y[m], cv=skf,
                              method="predict_proba")[:, 1]
        a = roc_auc_score(y[m], p)
        within[t] = a
        print(f"  {t:14s} within-task AUC = {a:.3f}")

    # ---- Cross-task grid, full vs scale-free features ----
    def transfer(names_subset, label, depth=10):
        idx = [names.index(n) for n in names_subset]
        print(f"\n[Transfer] {label} (depth={depth}):")
        res = {}
        for tr in np.unique(tasks):
            for te in np.unique(tasks):
                if tr == te:
                    continue
                m_tr, m_te = tasks == tr, tasks == te
                if y[m_tr].sum() < 20 or y[m_te].sum() < 20:
                    continue  # H3: skip statistically meaningless directions
                c = rf(depth)
                c.fit(X[np.ix_(m_tr, idx)], y[m_tr])
                a = roc_auc_score(y[m_te], c.predict_proba(X[np.ix_(m_te, idx)])[:, 1])
                res[(tr, te)] = a
        for (tr, te), a in res.items():
            w = within.get(te, float("nan"))
            print(f"  {tr:14s}->{te:14s} AUC={a:.3f}  "
                  f"(within-{te[:6]}={w:.3f}, gap={a-w:+.3f})")
        vals = list(res.values())
        print(f"  mean = {np.mean(vals):.3f}")
        return res

    res_full = transfer(names, "H1a: ALL features")
    res_sf = transfer(SCALE_FREE, "H1b: SCALE-FREE features only")
    res_shallow = transfer(names, "H1c: ALL features, shallow trees", depth=3)
    res_sf3 = transfer(SCALE_FREE, "H1d: SCALE-FREE + shallow", depth=3)

    # ---- H4: few-shot target calibration ----
    print("\n[H4] Few-shot calibration: source task + 50 target samples -> "
          "rest of target:")
    rng = np.random.default_rng(42)
    for tr in np.unique(tasks):
        for te in np.unique(tasks):
            if tr == te:
                continue
            m_tr, m_te = tasks == tr, tasks == te
            if y[m_tr].sum() < 20 or y[m_te].sum() < 60:
                continue
            te_idx = np.where(m_te)[0]
            cal = rng.choice(te_idx, size=50, replace=False)
            test = np.setdiff1d(te_idx, cal)
            Xc = np.vstack([X[m_tr], X[cal]])
            yc = np.concatenate([y[m_tr], y[cal]])
            c = rf()
            c.fit(Xc, yc)
            a = roc_auc_score(y[test], c.predict_proba(X[test])[:, 1])
            print(f"  {tr:14s}->{te:14s} +50cal AUC={a:.3f}  "
                  f"(zero-shot={res_full.get((tr, te), float('nan')):.3f}, "
                  f"within={within.get(te, float('nan')):.3f})")


if __name__ == "__main__":
    main()
