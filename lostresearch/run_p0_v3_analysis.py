"""
P0-1: Endpoint-conditioned null + P0-2: rank+CIS 联合判据 + P0-3: 正确样本对照

这三个实验都不需要重新 forward, 只用已保存的轨迹数据.

1. Endpoint null: 证明 88% 不是 selection bias
   - 构造保留最终 CIS<0 的 null (打乱非最终层)
   - 比较 null 下的 crossing rate

2. rank+CIS 联合: 同时报告 CIS>m AND rank<=k
   - 恢复 21-34% 作为主要结论
   - 做 (m, k) 敏感性

3. 正确样本对照: 比较 correct vs incorrect 的 crossing, dwell, peak rank
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def analyze_endpoint_null(all_results, n_null=1000):
    """P0-1: endpoint-conditioned null.

    错误样本的最终 CIS 必然 <0 (因为 generated 是 argmax).
    从这个条件出发, 构造 null: 保留最终层, 打乱中间层, 看 crossing rate.

    如果 null 的 crossing rate 也接近 88%, 那么 88% 就是 selection bias.
    如果 null 显著低于 88%, 那么 88% 有意义.
    """
    print("\n" + "=" * 70)
    print("P0-1: Endpoint-Conditioned Null")
    print("=" * 70)

    incorrect = [s for s in all_results if not s["final_correct"]]
    print(f"错误样本数: {len(incorrect)}")

    # 真实 crossing rate
    real_crossing = 0
    real_crossing_depths = []
    for s in incorrect:
        cis = s.get("cis", [])
        if len(cis) < 4:
            continue
        for i in range(1, len(cis)):
            if cis[i-1] > 0 and cis[i] < 0:
                real_crossing += 1
                real_crossing_depths.append(i / (len(cis)-1))
                break
    real_rate = 100 * real_crossing / len(incorrect)
    print(f"\n真实 crossing rate: {real_rate:.1f}% ({real_crossing}/{len(incorrect)})")
    if real_crossing_depths:
        print(f"  crossing depth: mean={np.mean(real_crossing_depths):.3f}, median={np.median(real_crossing_depths):.3f}")

    # Null 1: 在每个样本内打乱中间层 (保留最终层)
    np.random.seed(42)
    null_rates_1 = []
    for trial in range(10):  # 10 次重复
        null_crossing = 0
        for s in incorrect:
            cis = s.get("cis", []).copy()
            if len(cis) < 4:
                continue
            # 保留最终层, 打乱中间层
            mid = cis[:-1]
            np.random.shuffle(mid)
            shuffled = mid + [cis[-1]]
            for i in range(1, len(shuffled)):
                if shuffled[i-1] > 0 and shuffled[i] < 0:
                    null_crossing += 1
                    break
        null_rates_1.append(100 * null_crossing / len(incorrect))
    print(f"\nNull 1 (打乱中间层, 保留最终层): mean={np.mean(null_rates_1):.1f}% ± {np.std(null_rates_1):.1f}")

    # Null 2: 跨样本打乱 (每个位置随机选其他样本的该层 CIS, 但最终层保留)
    null_rates_2 = []
    for trial in range(10):
        null_crossing = 0
        for s in incorrect:
            cis = s.get("cis", [])
            if len(cis) < 4:
                continue
            # 最终层保留, 中间层从其他样本随机抽
            other_samples = np.random.choice(incorrect, len(cis)-1, replace=True)
            shuffled = [other_samples[j]["cis"][j] for j in range(len(cis)-1)]  # 近似
            shuffled.append(cis[-1])
            for i in range(1, len(shuffled)):
                if shuffled[i-1] > 0 and shuffled[i] < 0:
                    null_crossing += 1
                    break
        null_rates_2.append(100 * null_crossing / len(incorrect))
    print(f"Null 2 (跨样本打乱, 保留最终层): mean={np.mean(null_rates_2):.1f}% ± {np.std(null_rates_2):.1f}")

    # 判断
    print(f"\n--- 判断 ---")
    if real_rate > np.mean(null_rates_1) + 20:
        print(f"✓ 真实 crossing rate ({real_rate:.1f}%) 显著高于 null ({np.mean(null_rates_1):.1f}%)")
        print(f"  → 88% 不是纯 selection bias, 有真实信号")
    elif real_rate > np.mean(null_rates_1) + 10:
        print(f"? 真实 rate 略高于 null, 但差距不大")
    else:
        print(f"✗ 真实 rate 与 null 接近, 88% 可能是 selection bias")

    return {"real_rate": real_rate, "null_1": null_rates_1, "null_2": null_rates_2}


def analyze_rank_cis_joint(all_results):
    """P0-2: rank+CIS 联合判据 + 敏感性.

    同时要求:
      - CIS > m (信号有竞争力)
      - rank(y*) <= k (在词表中有竞争力)
      - 最终 CIS < 0 (信号丢失)

    恢复 21-34% 作为主要结论.
    """
    print("\n" + "=" * 70)
    print("P0-2: Rank+CIS Joint Criterion")
    print("=" * 70)

    incorrect = [s for s in all_results if not s["final_correct"]]
    print(f"错误样本数: {len(incorrect)}")

    # 不同 (m, k) 组合
    ms = [0.0, 0.5, 1.0, 2.0]
    ks = [1, 5, 10, 50]

    print(f"\n{'m\\k':<6}", end="")
    for k in ks:
        print(f"{k:<12}", end="")
    print()

    results = {}
    for m in ms:
        print(f"{m:<6}", end="")
        for k in ks:
            count = 0
            for s in incorrect:
                cis = s.get("cis", [])
                ranks = s.get("correct_rank", [])
                if len(cis) < 4 or len(ranks) < 4:
                    continue
                mid_cis = cis[1:-1]
                mid_ranks = ranks[1:-1]
                # 要求: 存在至少一层, CIS>m AND rank<=k
                has_competitive = any(
                    mid_cis[i] > m and mid_ranks[i] <= k
                    for i in range(len(mid_cis))
                )
                # 并且最终 CIS<0 (信号丢失)
                if has_competitive and cis[-1] < 0:
                    count += 1
            pct = 100 * count / len(incorrect)
            key = f"m{m}_k{k}"
            results[key] = {"m": m, "k": k, "count": count, "pct": pct}
            print(f"{pct:.1f}%      ", end="")
        print()

    # 重点: m=0, k=5 是之前 21-34% 的定义
    key_main = "m0.0_k5"
    if key_main in results:
        print(f"\n主要结论 (m=0, k=5): {results[key_main]['pct']:.1f}% ({results[key_main]['count']}/{len(incorrect)})")

    # Bootstrap CI for main
    m, k = 0.0, 5
    boots = []
    for _ in range(1000):
        sample = np.random.choice(incorrect, len(incorrect), replace=True)
        cnt = sum(1 for s in sample
                 if any(c > m and r <= k for c, r in zip(s["cis"][1:-1], s["correct_rank"][1:-1]))
                 and s["cis"][-1] < 0)
        boots.append(100 * cnt / len(sample))
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    print(f"  95% CI: [{ci_low:.1f}, {ci_high:.1f}]")

    return results


def analyze_correct_vs_incorrect(all_results):
    """P0-3: 正确样本对照.

    比较正确 vs 错误样本的:
    - crossing rate
    - dwell time (CIS>0 的持续层数)
    - peak rank
    - peak-to-final decay
    """
    print("\n" + "=" * 70)
    print("P0-3: Correct vs Incorrect Samples")
    print("=" * 70)

    correct = [s for s in all_results if s["final_correct"]]
    incorrect = [s for s in all_results if not s["final_correct"]]

    for label, group in [("Correct", correct), ("Incorrect", incorrect)]:
        if not group:
            continue
        print(f"\n  {label} (n={len(group)}):")

        # 1. Crossing rate (正→负)
        crossing = 0
        crossing_depths = []
        for s in group:
            cis = s.get("cis", [])
            if len(cis) < 4:
                continue
            for i in range(1, len(cis)):
                if cis[i-1] > 0 and cis[i] < 0:
                    crossing += 1
                    crossing_depths.append(i / (len(cis)-1))
                    break
        print(f"    正→负 crossing: {crossing}/{len(group)} ({100*crossing/len(group):.1f}%)")
        if crossing_depths:
            print(f"    crossing depth: mean={np.mean(crossing_depths):.3f}, median={np.median(crossing_depths):.3f}")

        # 2. Dwell time (CIS>0 的层数比例)
        dwell_times = []
        for s in group:
            cis = s.get("cis", [])
            if len(cis) < 4:
                continue
            pos_count = sum(1 for c in cis if c > 0)
            dwell_times.append(pos_count / len(cis))
        print(f"    CIS>0 dwell time: mean={np.mean(dwell_times):.3f}, median={np.median(dwell_times):.3f}")

        # 3. Peak rank (中间层最佳 rank)
        peak_ranks = []
        for s in group:
            ranks = s.get("correct_rank", [])
            if len(ranks) < 4:
                continue
            mid_ranks = ranks[1:-1]
            peak_ranks.append(min(mid_ranks))
        print(f"    peak rank: median={np.median(peak_ranks):.0f}, mean={np.mean(peak_ranks):.0f}")

        # 4. Peak-to-final CIS decay
        decays = []
        for s in group:
            cis = s.get("cis", [])
            if len(cis) < 4:
                continue
            peak = max(cis[1:-1])
            final = cis[-1]
            decays.append(peak - final)
        print(f"    peak-to-final decay: mean={np.mean(decays):.3f}, median={np.median(decays):.3f}")

    # 判断
    print(f"\n--- 判断 ---")
    correct_crossing = sum(1 for s in correct
                            if any(s["cis"][i-1] > 0 and s["cis"][i] < 0
                                   for i in range(1, len(s["cis"]))))
    incorrect_crossing = sum(1 for s in incorrect
                              if any(s["cis"][i-1] > 0 and s["cis"][i] < 0
                                     for i in range(1, len(s["cis"]))))
    cr = 100 * correct_crossing / len(correct) if correct else 0
    ir = 100 * incorrect_crossing / len(incorrect) if incorrect else 0
    if cr > 50:
        print(f"⚠ 正确样本也有 {cr:.1f}% crossing → crossing 不是错误特有")
    elif ir > cr + 20:
        print(f"✓ 错误样本 crossing ({ir:.1f}%) 显著高于正确样本 ({cr:.1f}%)")
    else:
        print(f"? 差异不大")


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        return

    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} samples")

    null_results = analyze_endpoint_null(all_results)
    joint_results = analyze_rank_cis_joint(all_results)
    correct_vs_incorrect = analyze_correct_vs_incorrect(all_results)

    out = {"endpoint_null": null_results,
            "rank_cis_joint": joint_results,
            "correct_vs_incorrect": "see log"}
    out_file = os.path.join(config.DATA_DIR, "p0_v3_analysis_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
