"""
P0 实验 1+2: Top-k sensitivity curve + Multi-token sequence-level CIS

这两个实验都不需要重新 forward:
- Top-k: 直接用已存的 correct_rank 统计不同 k 下的"信号丢失"占比
- Multi-token CIS: 需要重新 forward (teacher forcing), 单独脚本

这个脚本只做 Top-k sensitivity.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def analyze_topk_sensitivity(all_results):
    """对不同 k 值统计 '信号丢失' 占比, 证明结论不依赖阈值选择."""
    print("\n" + "=" * 70)
    print("P0-1: Top-k Sensitivity Analysis")
    print("=" * 70)

    ks = [1, 3, 5, 10, 20, 50]
    incorrect = [s for s in all_results if not s["final_correct"]]

    print(f"\n错误样本数: {len(incorrect)}")
    print(f"\n{'k':<6} {'信号丢失占比':<15} {'95% CI':<20}")
    print("-" * 45)

    results = {}
    for k in ks:
        lost_count = 0
        for s in incorrect:
            ranks = s["correct_rank"]
            if len(ranks) < 4:
                continue
            mid_ranks = ranks[1:-1]
            mid_best = min(mid_ranks)
            final_rank = ranks[-1]
            # 信号丢失: 中间层 rank <= k, 但最终层 rank > k*2
            if mid_best <= k and final_rank > k * 2:
                lost_count += 1

        pct = 100 * lost_count / len(incorrect)
        # Bootstrap CI
        boots = []
        for _ in range(1000):
            sample = np.random.choice(incorrect, len(incorrect), replace=True)
            cnt = sum(1 for s in sample
                     if min(s["correct_rank"][1:-1]) <= k
                     and s["correct_rank"][-1] > k * 2)
            boots.append(100 * cnt / len(sample))
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        results[k] = {"pct": pct, "ci_low": ci_low, "ci_high": ci_high}
        print(f"{k:<6} {pct:.1f}%           [{ci_low:.1f}, {ci_high:.1f}]")

    # 画图
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = [results[k]["pct"] for k in ks]
    ci_lows = [results[k]["ci_low"] for k in ks]
    ci_highs = [results[k]["ci_high"] for k in ks]
    ax.errorbar(ks, pcts, yerr=[np.array(pcts)-np.array(ci_lows),
                                  np.array(ci_highs)-np.array(pcts)],
                marker="o", linewidth=2, capsize=5)
    ax.set_xlabel("Threshold k (mid rank <= k)")
    ax.set_ylabel("'Signal Lost' Percentage")
    ax.set_title("Top-k Sensitivity: Signal Loss Rate vs Threshold")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(config.FIGURE_DIR, "topk_sensitivity_Qwen3-8B.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()

    # 判断
    pcts_arr = np.array(pcts)
    if pcts_arr.max() - pcts_arr.min() < 10:
        print(f"\n✓ 现象在 k=1..50 下稳定 (变化 < 10pp)")
    else:
        print(f"\n? 现象对 k 敏感 (变化 {pcts_arr.max()-pcts_arr.min():.1f}pp)")

    return results


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        return

    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} samples")

    k_results = analyze_topk_sensitivity(all_results)

    out_file = os.path.join(config.DATA_DIR, "topk_sensitivity_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({str(k): v for k, v in k_results.items()}, f, indent=2)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
