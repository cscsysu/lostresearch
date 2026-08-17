"""
Matched-null crossing analysis: stronger null than random competitor.

Reviewer concern: random competitor is naturally weaker (not argmax-selected),
so high crossing rate (99%) is expected. Need a null that controls for the
competitor's strength (final rank, logit, frequency).

Protocol:
- For each error, instead of using the actual final competitor or a random token,
  select a "matched" competitor: a token with similar final-layer rank/logit as
  the true final competitor, but drawn from a different semantic category.
- Compare crossing rates: observed vs rank-matched null vs random null.

If rank-matched null still shows high crossing rate -> crossing is truly trivial.
If rank-matched null shows lower rate -> original result was partially due to
competitor weakness.

Usage:
  python run_matched_null.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main():
    # Try multiple paths for results file
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt_path = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output", "outputs", "data", "full_results_Qwen3-8B.json")
        if os.path.exists(alt_path):
            results_file = alt_path
        else:
            print(f"ERROR: Cannot find results file at {results_file} or {alt_path}")
            return
    with open(results_file) as f:
        all_results = json.load(f)

    errors = [s for s in all_results if not s.get("final_correct")]
    print(f"Total errors: {len(errors)}")

    # Observed crossing rate: gold ever beats final competitor
    observed_crossings = 0
    for s in errors:
        cis = s.get("cis", [])
        if any(c > 0 for c in cis[:-1]):
            observed_crossings += 1
    observed_rate = observed_crossings / len(errors)
    print(f"\n[Observed] Gold ever beats final competitor: {observed_crossings}/{len(errors)} ({100*observed_rate:.1f}%)")

    # Random-competitor null (existing): use another error's generated token
    np.random.seed(42)
    n_perm = 200
    random_rates = []
    for _ in range(n_perm):
        shuffled = list(errors)
        np.random.shuffle(shuffled)
        crossings = 0
        for s, s_comp in zip(errors, shuffled):
            gold_lp = np.array(s["correct_logprob"])
            comp_gold_lp = np.array(s_comp["correct_logprob"])
            comp_cis = np.array(s_comp["cis"])
            comp_gen_lp = comp_gold_lp - comp_cis  # generated token logprob
            margin = gold_lp[:-1] - comp_gen_lp[:-1]
            if any(m > 0 for m in margin):
                crossings += 1
        random_rates.append(crossings / len(errors))
    random_mean = np.mean(random_rates)
    random_std = np.std(random_rates)
    print(f"\n[Random-competitor null] ({n_perm} permutations)")
    print(f"  Mean crossing rate: {100*random_mean:.1f}% ± {100*random_std:.1f}%")
    print(f"  95% CI: [{100*(random_mean-1.96*random_std):.1f}%, {100*(random_mean+1.96*random_std):.1f}%]")

    # Rank-matched null: use another error's generated token, but only if
    # the replacement has similar final-layer gen_logprob (within ±1 logit)
    # This controls for competitor strength
    print(f"\n[Rank-matched null] (matching competitor final logprob within ±1)")
    final_gen_lps = [s["correct_logprob"][-1] - s["cis"][-1] for s in errors]

    matched_rates = []
    for _ in range(min(50, n_perm)):
        np.random.shuffle(shuffled)
        crossings = 0
        matched = 0
        for i, s in enumerate(errors):
            # Find a random error with similar final gen logprob
            target_lp = final_gen_lps[i]
            candidates = [j for j in range(len(errors)) if j != i and
                         abs(final_gen_lps[j] - target_lp) < 1.0]
            if not candidates:
                # Fallback to any
                j = np.random.randint(len(errors))
            else:
                j = np.random.choice(candidates)
                matched += 1

            s_comp = errors[j]
            gold_lp = np.array(s["correct_logprob"])
            comp_gold_lp = np.array(s_comp["correct_logprob"])
            comp_cis = np.array(s_comp["cis"])
            comp_gen_lp = comp_gold_lp - comp_cis
            margin = gold_lp[:-1] - comp_gen_lp[:-1]
            if any(m > 0 for m in margin):
                crossings += 1
        matched_rates.append(crossings / len(errors))

    matched_mean = np.mean(matched_rates)
    matched_std = np.std(matched_rates)
    print(f"  Mean crossing rate: {100*matched_mean:.1f}% ± {100*matched_std:.1f}%")
    print(f"  Successfully matched: ~{matched}/{len(errors)} per permutation")

    # Summary
    print(f"\n{'='*60}")
    print(f"CROSSING RATE COMPARISON")
    print(f"{'='*60}")
    print(f"  Observed (gold vs actual final competitor): {100*observed_rate:.1f}%")
    print(f"  Random-competitor null:                     {100*random_mean:.1f}% ± {100*random_std:.1f}%")
    print(f"  Rank-matched null:                         {100*matched_mean:.1f}% ± {100*matched_std:.1f}%")
    print(f"\n  Interpretation:")
    if matched_mean > observed_rate:
        print(f"  Even rank-matched competitors show HIGHER crossing ({100*matched_mean:.0f}% > {100*observed_rate:.0f}%)")
        print(f"  -> Crossing is trivially weak regardless of competitor strength")
    else:
        print(f"  Rank-matched null is LOWER ({100*matched_mean:.0f}% < {100*random_mean:.0f}%)")
        print(f"  -> Competitor strength matters; original random null overestimates")

    out_file = os.path.join(config.DATA_DIR, "matched_null_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({
            "observed_rate": observed_rate,
            "random_null_mean": random_mean, "random_null_std": random_std,
            "matched_null_mean": matched_mean, "matched_null_std": matched_std,
        }, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
