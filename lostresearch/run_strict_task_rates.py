"""
Strict (Eq. 8) per-task preservation rates with bootstrap CIs.

Reviewer round-6 requirement: "preservation failures are a small minority"
was only shown on the Qwen3-8B calibrated subset under the strict criterion,
while the cross-task 5-28% numbers use the looser metric. The two must not
be conflated. This script reports the STRICT criterion per task with CIs.

Strict label (Eq. 8): exists intermediate layer with
    rank(y*) <= k  AND  CIS_comp > 0
and final layer CIS_comp < 0.

Usage:
  python run_strict_task_rates.py            # uses full_results
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def strict_label(s, k):
    ranks = s.get("correct_rank", [])
    cis = s.get("cis", [])
    if len(ranks) < 2 or len(cis) < 2 or len(ranks) != len(cis):
        return None
    inter = range(len(ranks) - 1)
    competitive = any(ranks[l] <= k and cis[l] > 0 for l in inter)
    return int(competitive and cis[-1] < 0)


def boot_ci(flags, n_boot=2000, seed=0):
    a = np.array(flags, dtype=float)
    if len(a) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    rates = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return (float(a.mean()), float(np.percentile(rates, 2.5)),
            float(np.percentile(rates, 97.5)))


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as fh:
        all_results = json.load(fh)

    k = config.RANK_COMPETITIVE
    errors = [s for s in all_results if not s.get("final_correct")]
    print(f"Errors: {len(errors)}  (strict criterion, k={k})\n")

    per_task = {}
    for s in errors:
        lab = strict_label(s, k)
        if lab is None:
            continue
        per_task.setdefault(s.get("task", "?"), []).append(lab)

    print(f"{'task':16s} {'n':>5s} {'rate':>7s} {'95% CI':>16s}")
    out = {}
    for t in sorted(per_task, key=lambda t: -np.mean(per_task[t])):
        rate, lo, hi = boot_ci(per_task[t])
        out[t] = {"n": len(per_task[t]), "rate": rate, "ci_lo": lo, "ci_hi": hi}
        print(f"{t:16s} {len(per_task[t]):5d} {100*rate:6.1f}% "
              f"[{100*lo:.1f}, {100*hi:.1f}]")

    all_flags = [f for v in per_task.values() for f in v]
    rate, lo, hi = boot_ci(all_flags)
    print(f"{'OVERALL':16s} {len(all_flags):5d} {100*rate:6.1f}% "
          f"[{100*lo:.1f}, {100*hi:.1f}]")
    out["_overall"] = {"n": len(all_flags), "rate": rate, "ci_lo": lo, "ci_hi": hi}

    out_file = os.path.join(config.DATA_DIR, "strict_task_rates_Qwen3-8B.json")
    with open(out_file, "w") as fh:
        json.dump({"k": k, "per_task": out}, fh, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
