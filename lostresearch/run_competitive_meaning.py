"""
Mechanism meaning of the "competitive" (rank<=k) definition.

Reviewer concern: is "rank <= k is competitive" just an empirical threshold,
or does it correspond to a real internal state of the model?

Claim we support: entering the top-k corresponds to a phase transition in the
representation -- the gold token's log-probability jumps from a noise-level
value (very negative) to a substantive value once it becomes competitive. The
top-k definition therefore marks the moment the gold answer emerges from the
vocabulary noise, which is why k=5 (or any small k) captures a real internal
state rather than being an arbitrary cut.

For each sample we measure, at the first layer where the gold token's rank
drops to <= k:
  - the gold log-probability just before entering (noise level)
  - the gold log-probability just after entering
  - the jump (post - pre)
A large positive jump in almost all samples shows the competitive definition
tracks a genuine representation transition.

Usage:
  python run_competitive_meaning.py --data outputs/data/full_results_Qwen3-8B.json
"""
import argparse
import json
import os
import sys

import numpy as np


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


def analyze(samples, k=5):
    pre_vals, post_vals, jumps, entered = [], [], [], 0
    for s in samples:
        ranks = s["correct_rank"]
        lp = s["correct_logprob"]
        first = next((i for i, r in enumerate(ranks) if r <= k), None)
        if first is None:
            continue
        entered += 1
        pre = lp[max(0, first - 3):first]
        post = lp[first:first + 3]
        pre_mean = float(np.mean(pre)) if len(pre) else 0.0
        post_mean = float(np.mean(post)) if len(post) else 0.0
        pre_vals.append(pre_mean)
        post_vals.append(post_mean)
        jumps.append(post_mean - pre_mean)
    return {
        "k": k,
        "n_entered": entered,
        "n_total": len(samples),
        "pre_median": float(np.median(pre_vals)),
        "post_median": float(np.median(post_vals)),
        "jump_median": float(np.median(jumps)),
        "jump_mean": float(np.mean(jumps)),
        "jump_positive_frac": float(np.mean([j > 0 for j in jumps])),
        "post_prob_median": float(np.exp(np.median(post_vals))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ks", default="1,5,10")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    samples = load_samples(args.data)
    print(f"Loaded {len(samples)} samples")
    for k in [int(x) for x in args.ks.split(",")]:
        r = analyze(samples, k)
        print(f"\n=== k={k} ===")
        print(f"  entered competitive: {r['n_entered']}/{r['n_total']}")
        print(f"  gold logprob before entering: median={r['pre_median']:.2f} "
              f"(prob ~{np.exp(r['pre_median']):.1e})")
        print(f"  gold logprob after entering:  median={r['post_median']:.2f} "
              f"(prob ~{r['post_prob_median']:.3f})")
        print(f"  jump (post-pre): median={r['jump_median']:.2f}, "
              f"mean={r['jump_mean']:.2f}")
        print(f"  fraction with positive jump: {100*r['jump_positive_frac']:.1f}%")

    out = args.out
    if out:
        with open(out, "w") as f:
            json.dump({f"k{k}": analyze(samples, k)
                       for k in [int(x) for x in args.ks.split(",")]}, f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
