"""
Final-layer confidence baseline for decay prediction (reviewer 3.4).

The reviewer asks: if the model's own final-layer confidence already predicts
correctness, what does the trajectory add? This script compares decay
prediction from (a) final-layer confidence signals (margin, rank, entropy)
against (b) the first-half trajectory features.

Direction-agnostic AUC is reported for each single feature (tries both
polarities and keeps the better one), so no sign assumption is baked in.

Usage:
  python run_final_layer_baseline.py --data <trajectory.json> [--out out.json]
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import config
from prediction import extract_targets
from run_p0_baselines import compute_entropy_from_top5, extract_strong_features


def load_samples(path):
    if not os.path.exists(path):
        sys.exit(f"  ! file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "trajectory_results" in data:
        return data["trajectory_results"]
    if isinstance(data, list):
        return data
    sys.exit(f"  ! unrecognized format: {path}")


def dir_agnostic_auc(y, feat):
    """Report the AUC using whichever polarity is informative (>= 0.5)."""
    a1 = roc_auc_score(y, feat)
    a2 = roc_auc_score(y, -feat)
    return max(a1, a2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    samples = load_samples(args.data)

    # Build aligned feature matrix + labels using the same code path as the
    # strong-baseline / cross-task experiments (scale-invariant trajectory feats).
    feats, y = [], []
    valid_idx = []
    for i, s in enumerate(samples):
        f = extract_strong_features(s, config.PREDICTION_T0)
        t = extract_targets(s)
        if f and t:
            valid_idx.append(i)
            feats.append(f)
            y.append(t["will_decay"])
    y = np.array(y)
    feature_names = list(feats[0].keys())

    # Final-layer confidence signals (from the raw trajectory fields).
    final_margin = np.array([s["cis"][-1] for s in samples])[valid_idx]
    final_rank = np.array([s["correct_rank"][-1] for s in samples])[valid_idx]
    final_lp = np.array([s["correct_logprob"][-1] for s in samples])[valid_idx]
    final_entropy = np.array(
        [compute_entropy_from_top5(s.get("top5", [])) for s in samples]
    )[valid_idx]

    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    print("\n=== Final-layer confidence baselines (direction-agnostic AUC) ===")
    single = {
        "final margin": dir_agnostic_auc(y, final_margin),
        "final rank": dir_agnostic_auc(y, final_rank),
        "final logprob": dir_agnostic_auc(y, final_lp),
        "final entropy": dir_agnostic_auc(y, final_entropy),
    }
    for k, v in single.items():
        print(f"  {k:<16} AUC = {v:.3f}")

    # Combined final-layer confidence model (CV).
    X_final = np.stack([final_margin, final_rank, final_entropy], axis=1)
    X_fs = StandardScaler().fit_transform(X_final)
    auc_final = cross_val_score(
        LogisticRegression(max_iter=2000), X_fs, y, cv=skf, scoring="roc_auc"
    )
    print(f"  final [margin+rank+entropy] CV AUC = {auc_final.mean():.3f} ± {auc_final.std():.3f}")

    # First-half trajectory (same 8 features as strong baselines), 5-fold CV.
    X_traj = np.array([[f[k] for k in feature_names] for f in feats])
    X_ts = StandardScaler().fit_transform(X_traj)
    auc_traj = cross_val_score(
        LogisticRegression(max_iter=2000), X_ts, y, cv=skf, scoring="roc_auc"
    )
    print(f"  first-half trajectory (8 feats) CV AUC = {auc_traj.mean():.3f} ± {auc_traj.std():.3f}")

    gain = auc_traj.mean() - auc_final.mean()
    print(f"\n  trajectory - final-confidence gain = {gain:+.3f} AUC")

    results = {
        "n": len(y),
        "decay_rate": float(y.mean()),
        "final_layer_single": single,
        "final_layer_combined_cv": {
            "mean": float(auc_final.mean()), "std": float(auc_final.std()),
        },
        "trajectory_cv": {
            "mean": float(auc_traj.mean()), "std": float(auc_traj.std()),
        },
        "gain": float(gain),
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  saved: {args.out}")
    return results


if __name__ == "__main__":
    main()
