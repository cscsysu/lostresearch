"""
Taxonomy robustness: is the formation/preservation split an artifact of
the threshold, the decoder, or the answer-length distribution?

Runs four robustness checks, all with bootstrap 95% CIs:
  A. Threshold sensitivity: k in {1,3,5,10,20,50}
  B. Answer-length stratification: single-token vs multi-token answers
  C. Per-task rates with CIs (addresses "no CIs reported")
  D. Definition sensitivity: strict (Eq.8) vs broad (signal-loss) criteria

Usage:
  python run_taxonomy_robustness.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def bootstrap_ci(flags, n_boot=2000, seed=42):
    """Bootstrap 95% CI for a proportion."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(flags, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rate = arr.mean()
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        boots.append(arr[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rate, lo, hi


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as fh:
        all_results = json.load(fh)

    errors = [s for s in all_results if not s.get("final_correct")]
    print(f"Total errors: {len(errors)}\n")

    out = {}

    # ---------- A. Threshold sensitivity with CIs ----------
    print("=" * 70)
    print("A. THRESHOLD SENSITIVITY (strict criterion: rank<=k AND CIS>0, final CIS<0)")
    print("=" * 70)
    out["threshold"] = {}
    for k in [1, 3, 5, 10, 20, 50]:
        flags = []
        for s in errors:
            ranks = s.get("correct_rank", [])
            cis = s.get("cis", [])
            if len(ranks) < 2 or len(cis) < 2:
                continue
            inter_ok = any(ranks[i] <= k and cis[i] > 0 for i in range(len(ranks) - 1))
            final_neg = cis[-1] < 0
            flags.append(int(inter_ok and final_neg))
        rate, lo, hi = bootstrap_ci(flags)
        print(f"  k={k:2d}: {100*rate:5.1f}%  95% CI [{100*lo:4.1f}%, {100*hi:4.1f}%]  (n={len(flags)})")
        out["threshold"][k] = {"rate": rate, "ci_lo": lo, "ci_hi": hi, "n": len(flags)}

    # ---------- B. Answer-length stratification ----------
    print("\n" + "=" * 70)
    print("B. ANSWER-LENGTH STRATIFICATION (does multi-token bias the rate?)")
    print("=" * 70)
    out["length"] = {}
    for label, cond in [
        ("1 word", lambda s: len(s.get("answer", "").split()) == 1),
        ("2 words", lambda s: len(s.get("answer", "").split()) == 2),
        ("3+ words", lambda s: len(s.get("answer", "").split()) >= 3),
    ]:
        subset = [s for s in errors if cond(s)]
        flags = []
        for s in subset:
            ranks = s.get("correct_rank", [])
            if len(ranks) < 2:
                continue
            flags.append(int(min(ranks[:-1]) <= config.RANK_COMPETITIVE))
        if flags:
            rate, lo, hi = bootstrap_ci(flags)
            print(f"  {label:9s}: {100*rate:5.1f}%  95% CI [{100*lo:4.1f}%, {100*hi:4.1f}%]  (n={len(flags)})")
            out["length"][label] = {"rate": rate, "ci_lo": lo, "ci_hi": hi, "n": len(flags)}

    # ---------- C. Per-task rates with CIs ----------
    print("\n" + "=" * 70)
    print("C. PER-TASK RATES WITH BOOTSTRAP 95% CI")
    print("=" * 70)
    out["per_task"] = {}
    tasks = sorted(set(s.get("task", "unknown") for s in errors))
    for task in tasks:
        subset = [s for s in errors if s.get("task") == task]
        flags = []
        for s in subset:
            ranks = s.get("correct_rank", [])
            if len(ranks) < 2:
                continue
            # broad signal-loss criterion (matches Table in paper)
            flags.append(int(min(ranks[:-1]) <= 5 and ranks[-1] > 10))
        if flags:
            rate, lo, hi = bootstrap_ci(flags)
            print(f"  {task:16s}: {100*rate:5.1f}%  95% CI [{100*lo:4.1f}%, {100*hi:4.1f}%]  (n={len(flags)})")
            out["per_task"][task] = {"rate": rate, "ci_lo": lo, "ci_hi": hi, "n": len(flags)}

    # ---------- D. Definition sensitivity ----------
    print("\n" + "=" * 70)
    print("D. DEFINITION SENSITIVITY (strict Eq.8 vs broad signal-loss)")
    print("=" * 70)
    strict, broad, either = [], [], []
    for s in errors:
        ranks = s.get("correct_rank", [])
        cis = s.get("cis", [])
        if len(ranks) < 2 or len(cis) < 2:
            continue
        st = int(any(ranks[i] <= 5 and cis[i] > 0 for i in range(len(ranks) - 1)) and cis[-1] < 0)
        br = int(min(ranks[:-1]) <= 5 and ranks[-1] > 10)
        strict.append(st)
        broad.append(br)
        either.append(int(st or br))
    for nm, flags in [("Strict (Eq.8)", strict), ("Broad (signal-loss)", broad),
                      ("Either", either)]:
        rate, lo, hi = bootstrap_ci(flags)
        print(f"  {nm:22s}: {100*rate:5.1f}%  95% CI [{100*lo:4.1f}%, {100*hi:4.1f}%]")
    # agreement
    agree = sum(1 for a, b in zip(strict, broad) if a == b) / len(strict)
    both = sum(1 for a, b in zip(strict, broad) if a == 1 and b == 1)
    print(f"\n  Agreement between definitions: {100*agree:.1f}%")
    print(f"  Flagged by BOTH: {both} samples")
    out["definition"] = {
        "strict": bootstrap_ci(strict), "broad": bootstrap_ci(broad),
        "agreement": agree, "both": both,
    }

    out_file = os.path.join(config.DATA_DIR, "taxonomy_robustness_Qwen3-8B.json")
    with open(out_file, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
