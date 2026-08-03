"""Analysis: 信号丢失判据 + 统计 + 可视化."""
import json
import os
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def classify_sample(sample: Dict) -> str:
    """把样本分到 5 类之一 (基于 CIS 轨迹).

    返回:
        - "Absent": 正确答案从未有竞争力 (中间层最佳 rank > 50)
        - "Late-Emergent": 后半段才出现信号 (SEL >= 0.65)
        - "Early-Decay": 前半段信号有竞争力, 但最终丢失
        - "Gradual-Buildup": 信号逐渐增强
        - "Fluctuating": 其他
    """
    ranks = sample["correct_rank"]
    n = len(ranks)
    if n < 4:
        return "Absent"

    # 中间层最佳 rank (排除第一层和最后一层)
    mid_ranks = ranks[1:-1]
    mid_best_rank = min(mid_ranks)
    final_rank = ranks[-1]

    # SEL: 首次达到 rank <= 5 的相对深度
    sel = None
    for i, r in enumerate(ranks):
        if r <= config.RANK_COMPETITIVE:
            sel = i / (n - 1)
            break

    if mid_best_rank > 50:
        return "Absent"
    if sel is not None and sel >= 0.65:
        return "Late-Emergent"
    if mid_best_rank <= config.RANK_COMPETITIVE and final_rank > config.RANK_LOST:
        return "Early-Decay"
    # 检查是否单调递增 (Gradual-Buildup)
    first_half = ranks[:n//2]
    second_half = ranks[n//2:]
    if np.median(second_half) < np.median(first_half):
        return "Gradual-Buildup"
    return "Fluctuating"


def compute_trajectory_metrics(sample: Dict) -> Dict:
    """计算单个样本的轨迹指标."""
    cis = sample.get("cis", [])
    ranks = sample["correct_rank"]
    logprobs = sample["correct_logprob"]
    n = len(ranks)

    if n < 2:
        return {}

    # SEL: 首次达到 rank <= 5 的相对深度
    sel = 1.0
    for i, r in enumerate(ranks):
        if r <= config.RANK_COMPETITIVE:
            sel = i / (n - 1)
            break

    # 中间层最佳
    mid_best_rank = min(ranks[1:-1]) if n > 2 else min(ranks)
    mid_best_logprob = max(logprobs[1:-1]) if n > 2 else max(logprobs)

    # 最终层
    final_rank = ranks[-1]
    final_logprob = logprobs[-1]

    # 是否信号丢失 (基于 CIS: 中间层 CIS 有竞争力, 但最终层 CIS<0)
    signal_lost = (mid_best_rank <= config.RANK_COMPETITIVE
                   and cis and cis[-1] < 0)

    # CIS 变化
    if cis:
        cis_final = cis[-1]
        cis_max_mid = max(cis[1:-1]) if len(cis) > 2 else max(cis)
        cis_delta = cis_max_mid - cis_final
    else:
        cis_final = cis_max_mid = cis_delta = 0

    return {
        "SEL": sel,
        "mid_best_rank": mid_best_rank,
        "mid_best_logprob": mid_best_logprob,
        "final_rank": final_rank,
        "final_logprob": final_logprob,
        "signal_lost": signal_lost,
        "pattern": classify_sample(sample),
        "cis_final": cis_final,
        "cis_max_mid": cis_max_mid,
        "cis_delta": cis_delta,
    }


def print_summary(all_samples: List[Dict]):
    """打印统计摘要."""
    print("\n" + "=" * 70)
    print("FULL EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(all_samples)}")

    correct = [s for s in all_samples if s["final_correct"]]
    incorrect = [s for s in all_samples if not s["final_correct"]]
    print(f"Correct:   {len(correct)} ({100*len(correct)/max(len(all_samples),1):.1f}%)")
    print(f"Incorrect: {len(incorrect)} ({100*len(incorrect)/max(len(all_samples),1):.1f}%)")

    # 计算每个样本的指标
    for s in all_samples:
        s["metrics"] = compute_trajectory_metrics(s)

    # 按任务分组
    tasks = set(s["task"] for s in all_samples)
    for task in sorted(tasks):
        task_samples = [s for s in all_samples if s["task"] == task]
        task_correct = [s for s in task_samples if s["final_correct"]]
        task_incorrect = [s for s in task_samples if not s["final_correct"]]
        print(f"\n--- {task} (n={len(task_samples)}) ---")
        print(f"  Correct: {len(task_correct)}, Incorrect: {len(task_incorrect)}")

        if task_incorrect:
            # 信号丢失统计
            lost = sum(1 for s in task_incorrect if s["metrics"]["signal_lost"])
            print(f"  信号丢失 (mid rank<=5, final rank>10): "
                  f"{lost}/{len(task_incorrect)} ({100*lost/len(task_incorrect):.1f}%)")

            # CIS 负值 (错误答案信号压过正确答案)
            cis_neg = sum(1 for s in task_incorrect if s["metrics"]["cis_final"] < 0)
            print(f"  最终层 CIS < 0 (错误信号压过正确): "
                  f"{cis_neg}/{len(task_incorrect)} ({100*cis_neg/len(task_incorrect):.1f}%)")

            # 模式分布
            patterns = [s["metrics"]["pattern"] for s in task_incorrect]
            for p in ["Absent", "Early-Decay", "Late-Emergent", "Gradual-Buildup", "Fluctuating"]:
                cnt = patterns.count(p)
                print(f"    {p}: {cnt} ({100*cnt/len(task_incorrect):.1f}%)")

    # 总体统计
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if incorrect:
        lost_total = sum(1 for s in incorrect if s["metrics"]["signal_lost"])
        pct = 100 * lost_total / len(incorrect)
        print(f"1. 信号丢失 (mid rank<=5, final rank>10): "
              f"{lost_total}/{len(incorrect)} ({pct:.1f}%)")

        cis_neg_total = sum(1 for s in incorrect if s["metrics"]["cis_final"] < 0)
        pct2 = 100 * cis_neg_total / len(incorrect)
        print(f"2. 最终层 CIS<0 (错误信号压过正确): "
              f"{cis_neg_total}/{len(incorrect)} ({pct2:.1f}%)")

        # 真正的 "知道但没说": 中间层 rank<=5 且最终层 CIS<0
        know_but_lost = sum(1 for s in incorrect
                             if s["metrics"]["mid_best_rank"] <= 5
                             and s["metrics"]["cis_final"] < 0)
        pct3 = 100 * know_but_lost / len(incorrect)
        print(f"3. '知道但没说' (mid rank<=5 + final CIS<0): "
              f"{know_but_lost}/{len(incorrect)} ({pct3:.1f}%)")

        print()
        if pct3 > 30:
            print(f"  ✓ {pct3:.1f}% 的错误样本属于 '知道但没说'")
            print("  → 现象存在, 值得深入研究")
        elif pct3 > 10:
            print(f"  ? {pct3:.1f}% 的错误样本属于 '知道但没说', 处于临界区")
            print("  → 建议扩大样本量")
        else:
            print(f"  ✗ 仅 {pct3:.1f}% 的错误样本属于 '知道但没说'")
            print("  → 现象不显著, 需要重新审视假设")

    print("=" * 70)


def plot_trajectory_comparison(all_samples: List[Dict], save_path: str):
    """画正确 vs 错误的轨迹对比图 (3 列: correct_logprob, rank, CIS)."""
    correct = [s for s in all_samples if s["final_correct"]]
    incorrect = [s for s in all_samples if not s["final_correct"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    max_len = max(len(s["correct_logprob"]) for s in all_samples)

    def to_array(samples, key):
        arr = np.full((len(samples), max_len), np.nan)
        for i, s in enumerate(samples):
            vals = s[key]
            arr[i, :len(vals)] = vals
        return arr

    # 1. Correct log-prob
    ax = axes[0]
    if correct:
        arr = to_array(correct, "correct_logprob")
        for row in arr:
            ax.plot(range(max_len), row, color="green", alpha=0.2, linewidth=0.8)
        ax.plot(range(max_len), np.nanmean(arr, axis=0),
                color="green", linewidth=3, label=f"Correct (n={len(correct)})")
    if incorrect:
        arr = to_array(incorrect, "correct_logprob")
        for row in arr:
            ax.plot(range(max_len), row, color="red", alpha=0.3, linewidth=0.8)
        ax.plot(range(max_len), np.nanmean(arr, axis=0),
                color="red", linewidth=3, label=f"Incorrect (n={len(incorrect)})")
    ax.set_xlabel("Layer"); ax.set_ylabel("Log Prob")
    ax.set_title("Correct Answer Log Prob"); ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Rank
    ax = axes[1]
    if correct:
        arr = to_array(correct, "correct_rank")
        for row in arr:
            ax.plot(range(max_len), row, color="green", alpha=0.2, linewidth=0.8)
        ax.plot(range(max_len), np.nanmedian(arr, axis=0),
                color="green", linewidth=3, label=f"Correct (median)")
    if incorrect:
        arr = to_array(incorrect, "correct_rank")
        for row in arr:
            ax.plot(range(max_len), row, color="red", alpha=0.3, linewidth=0.8)
        ax.plot(range(max_len), np.nanmedian(arr, axis=0),
                color="red", linewidth=3, label=f"Incorrect (median)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Rank")
    ax.set_title("Correct Answer Rank"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. CIS (correct - generated)
    ax = axes[2]
    if correct:
        arr = to_array(correct, "cis")
        for row in arr:
            ax.plot(range(max_len), row, color="green", alpha=0.2, linewidth=0.8)
        ax.plot(range(max_len), np.nanmean(arr, axis=0),
                color="green", linewidth=3, label=f"Correct (n={len(correct)})")
    if incorrect:
        arr = to_array(incorrect, "cis")
        for row in arr:
            ax.plot(range(max_len), row, color="red", alpha=0.3, linewidth=0.8)
        ax.plot(range(max_len), np.nanmean(arr, axis=0),
                color="red", linewidth=3, label=f"Incorrect (n={len(incorrect)})")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("CIS (correct - generated)")
    ax.set_title("Correct Information Signal (CIS)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_single_trajectory(sample: Dict, save_path: str):
    """画单个样本的完整轨迹图 (3 列)."""
    layers = list(range(len(sample["correct_logprob"])))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Log-prob: correct vs generated
    ax = axes[0]
    ax.plot(layers, sample["correct_logprob"], "g-o", markersize=3, label="Correct Answer")
    ax.plot(layers, sample["generated_logprob"], "r-o", markersize=3, label="Generated Answer")
    ax.set_xlabel("Layer"); ax.set_ylabel("Log Prob")
    ax.set_title(f"Log Prob\nQ: {sample['question'][:40]}...")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Rank
    ax = axes[1]
    ax.plot(layers, sample["correct_rank"], "g-o", markersize=3, label="Correct Rank")
    ax.plot(layers, sample["generated_rank"], "r-o", markersize=3, label="Generated Rank")
    ax.set_xlabel("Layer"); ax.set_ylabel("Rank (log scale)")
    ax.set_title(f"Rank\nA: {sample['answer']} | Gen: {sample['generated'][:20]}")
    ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)

    # 3. CIS
    ax = axes[2]
    ax.plot(layers, sample["cis"], "b-o", markersize=3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("CIS")
    status = "✓ Correct" if sample["final_correct"] else "✗ Incorrect"
    ax.set_title(f"CIS (correct - generated)\n{status}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_pattern_distribution(all_samples: List[Dict], save_path: str):
    """画 pattern 分布图 (按任务和正确/错误分组)."""
    tasks = sorted(set(s["task"] for s in all_samples))
    patterns = ["Absent", "Early-Decay", "Late-Emergent", "Gradual-Buildup", "Fluctuating"]

    fig, axes = plt.subplots(1, len(tasks) + 1, figsize=(5*(len(tasks)+1), 5))

    # 总体
    ax = axes[-1]
    incorrect = [s for s in all_samples if not s["final_correct"]]
    counts = [sum(1 for s in incorrect if s.get("metrics", {}).get("pattern") == p)
              for p in patterns]
    ax.bar(patterns, counts, color="salmon")
    ax.set_title(f"All Incorrect (n={len(incorrect)})")
    ax.tick_params(axis="x", rotation=45)

    # 按任务
    for i, task in enumerate(tasks):
        ax = axes[i]
        task_inc = [s for s in all_samples if s["task"] == task and not s["final_correct"]]
        counts = [sum(1 for s in task_inc if s.get("metrics", {}).get("pattern") == p)
                  for p in patterns]
        ax.bar(patterns, counts, color="salmon")
        ax.set_title(f"{task} Incorrect (n={len(task_inc)})")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()
