"""
P0 实验 2: Top-k / Threshold Sensitivity curve

两种判据:
1. CIS-based (主): 中间 CIS peak > threshold, 最终 CIS < 0
2. Rank-based (辅): 中间 rank <= k, 最终 rank > k*2
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def analyze_sensitivity(all_results):
    """对不同阈值统计 '信号丢失' 占比, 证明结论不依赖阈值选择."""
    print("\n" + "=" * 70)
    print("P0-2: Threshold Sensitivity Analysis")
    print("=" * 70)

    incorrect = [s for s in all_results if not s["final_correct"]]
    print(f"\n错误样本数: {len(incorrect)}")

    # === 判据 1: CIS-based (主) ===
    print(f"\n--- 判据 1: CIS-based (主) ---")
    cis_thresholds = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    print(f"{'threshold':<12} {'信号丢失占比':<15} {'95% CI':<20}")
    print("-" * 47)

    cis_results = {}
    for thresh in cis_thresholds:
        lost_count = 0
        for s in incorrect:
            cis = s.get("cis", [])
            if len(cis) < 4:
                continue
            mid_cis = cis[1:-1]
            mid_max = max(mid_cis)
            final_cis = cis[-1]
            if mid_max > thresh and final_cis < 0:
                lost_count += 1

        pct = 100 * lost_count / len(incorrect)
        boots = []
        for _ in range(1000):
            sample = np.random.choice(incorrect, len(incorrect), replace=True)
            cnt = sum(1 for s in sample
                     if max(s.get("cis", [0])[1:-1]) > thresh
                     and s.get("cis", [0])[-1] < 0)
            boots.append(100 * cnt / len(sample))
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        cis_results[thresh] = {"pct": pct, "ci_low": ci_low, "ci_high": ci_high}
        print(f"{thresh:<12} {pct:.1f}%           [{ci_low:.1f}, {ci_high:.1f}]")

    # === 判据 2: Rank-based (辅) ===
    print(f"\n--- 判据 2: Rank-based (辅) ---")
    ks = [1, 3, 5, 10, 20, 50]
    print(f"{'k':<6} {'信号丢失占比':<15} {'95% CI':<20}")
    print("-" * 41)

    rank_results = {}
    for k in ks:
        lost_count = 0
        for s in incorrect:
            ranks = s["correct_rank"]
            if len(ranks) < 4:
                continue
            mid_best = min(ranks[1:-1])
            final_rank = ranks[-1]
            if mid_best <= k and final_rank > k * 2:
                lost_count += 1

        pct = 100 * lost_count / len(incorrect)
        boots = []
        for _ in range(1000):
            sample = np.random.choice(incorrect, len(incorrect), replace=True)
            cnt = sum(1 for s in sample
                     if min(s["correct_rank"][1:-1]) <= k
                     and s["correct_rank"][-1] > k * 2)
            boots.append(100 * cnt / len(sample))
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        rank_results[k] = {"pct": pct, "ci_low": ci_low, "ci_high": ci_high}
        print(f"{k:<6} {pct:.1f}%           [{ci_low:.1f}, {ci_high:.1f}]")

    # 画图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    pcts = [cis_results[t]["pct"] for t in cis_thresholds]
    ci_lows = [cis_results[t]["ci_low"] for t in cis_thresholds]
    ci_highs = [cis_results[t]["ci_high"] for t in cis_thresholds]
    ax1.errorbar(cis_thresholds, pcts,
                  yerr=[np.array(pcts)-np.array(ci_lows),
                        np.array(ci_highs)-np.array(pcts)],
                  marker="o", linewidth=2, capsize=5, color="blue")
    ax1.set_xlabel("CIS threshold (mid peak > threshold)")
    ax1.set_ylabel("'Signal Lost' Percentage")
    ax1.set_title("CIS-based Sensitivity (Main)")
    ax1.grid(True, alpha=0.3)

    pcts2 = [rank_results[k]["pct"] for k in ks]
    ci_lows2 = [rank_results[k]["ci_low"] for k in ks]
    ci_highs2 = [rank_results[k]["ci_high"] for k in ks]
    ax2.errorbar(ks, pcts2,
                  yerr=[np.array(pcts2)-np.array(ci_lows2),
                        np.array(ci_highs2)-np.array(pcts2)],
                  marker="s", linewidth=2, capsize=5, color="red")
    ax2.set_xlabel("Threshold k (mid rank <= k)")
    ax2.set_ylabel("'Signal Lost' Percentage")
    ax2.set_title("Rank-based Sensitivity (Aux)")
    ax2.set_xscale("log")
    ax2.set_xticks(ks)
    ax2.set_xticklabels(ks)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.FIGURE_DIR, "topk_sensitivity_Qwen3-8B.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()

    # 判断 (基于 CIS)
    pcts_arr = np.array(pcts)
    if pcts_arr.max() - pcts_arr.min() < 15:
        print(f"\n✓ CIS-based 现象稳定 (变化 < 15pp)")
    else:
        print(f"\n? CIS-based 现象对阈值敏感 (变化 {pcts_arr.max()-pcts_arr.min():.1f}pp)")

    return {"cis_based": cis_results, "rank_based": rank_results}


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        return

    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} samples")

    results = analyze_sensitivity(all_results)

    out_file = os.path.join(config.DATA_DIR, "topk_sensitivity_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
