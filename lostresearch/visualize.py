"""Visualization: 画 CIS 轨迹图, 正确 vs 错误对比图."""
import json
import os
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt
import numpy as np

import config


def plot_single_trajectory(sample: Dict, save_path: str = None):
    """画单个样本的 CIS 轨迹图."""
    logprobs = sample["correct_token_logprob"]
    ranks = sample["correct_token_rank"]
    layers = list(range(len(logprobs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: log probability
    ax1.plot(layers, logprobs, "b-o", markersize=3, linewidth=1.5)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Log Probability")
    ax1.set_title(f"Correct Answer LogProb\nQ: {sample['question'][:50]}...")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.log(0.1), color="r", linestyle="--", alpha=0.5, label="prob=0.1")
    ax1.legend()

    # 右图: rank (log scale)
    ax2.plot(layers, ranks, "g-o", markersize=3, linewidth=1.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Rank (log scale)")
    ax2.set_title(f"Correct Answer Rank\nA: {sample['answer']}")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # rank 越小越好, 反转 y 轴

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_correct_vs_incorrect(all_samples: List[Dict], save_path: str = None):
    """画正确样本 vs 错误样本的轨迹对比图."""
    correct_samples = [s for s in all_samples if s["final_correct"]]
    incorrect_samples = [s for s in all_samples if not s["final_correct"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: log prob 轨迹
    ax = axes[0]
    for s in correct_samples:
        n = len(s["correct_token_logprob"])
        ax.plot(range(n), s["correct_token_logprob"],
                color="green", alpha=0.3, linewidth=1)
    for s in incorrect_samples:
        n = len(s["correct_token_logprob"])
        ax.plot(range(n), s["correct_token_logprob"],
                color="red", alpha=0.5, linewidth=1)

    # 画平均轨迹
    max_len = max(len(s["correct_token_logprob"]) for s in all_samples)
    correct_arr = np.full((len(correct_samples), max_len), np.nan)
    incorrect_arr = np.full((len(incorrect_samples), max_len), np.nan)
    for i, s in enumerate(correct_samples):
        correct_arr[i, :len(s["correct_token_logprob"])] = s["correct_token_logprob"]
    for i, s in enumerate(incorrect_samples):
        incorrect_arr[i, :len(s["correct_token_logprob"])] = s["correct_token_logprob"]

    if len(correct_samples) > 0:
        ax.plot(range(max_len), np.nanmean(correct_arr, axis=0),
                color="green", linewidth=3, label=f"Correct (n={len(correct_samples)})")
    if len(incorrect_samples) > 0:
        ax.plot(range(max_len), np.nanmean(incorrect_arr, axis=0),
                color="red", linewidth=3, label=f"Incorrect (n={len(incorrect_samples)})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Log Probability of Correct Answer")
    ax.set_title("CIS Trajectory: Correct vs Incorrect")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图: rank 轨迹
    ax = axes[1]
    correct_arr_rank = np.full((len(correct_samples), max_len), np.nan)
    incorrect_arr_rank = np.full((len(incorrect_samples), max_len), np.nan)
    for i, s in enumerate(correct_samples):
        correct_arr_rank[i, :len(s["correct_token_rank"])] = s["correct_token_rank"]
    for i, s in enumerate(incorrect_samples):
        incorrect_arr_rank[i, :len(s["correct_token_rank"])] = s["correct_token_rank"]

    if len(correct_samples) > 0:
        ax.plot(range(max_len), np.nanmean(correct_arr_rank, axis=0),
                color="green", linewidth=3, label=f"Correct (n={len(correct_samples)})")
    if len(incorrect_samples) > 0:
        ax.plot(range(max_len), np.nanmean(incorrect_arr_rank, axis=0),
                color="red", linewidth=3, label=f"Incorrect (n={len(incorrect_samples)})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank of Correct Answer")
    ax.set_title("Rank Trajectory: Correct vs Incorrect")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def print_summary(all_samples: List[Dict]):
    """打印统计摘要."""
    print("\n" + "=" * 60)
    print("PILOT EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(all_samples)}")

    correct = [s for s in all_samples if s["final_correct"]]
    incorrect = [s for s in all_samples if not s["final_correct"]]
    print(f"Correct:   {len(correct)} ({100*len(correct)/len(all_samples):.1f}%)")
    print(f"Incorrect: {len(incorrect)} ({100*len(incorrect)/len(all_samples):.1f}%)")

    if not incorrect:
        print("\nNo incorrect samples, cannot analyze 'lost signal'.")
        return

    # 核心统计: 错误样本中, 中间层信号峰值高于最终层的占比
    print("\n--- 核心统计: 错误样本中的 '信号丢失' 现象 ---")
    signal_appeared_but_lost = 0
    for s in incorrect:
        logprobs = s["correct_token_logprob"]
        if len(logprobs) < 2:
            continue
        max_mid = max(logprobs[:-1])  # 中间层最大值
        final = logprobs[-1]         # 最终层
        # 如果中间层信号明显高于最终层 (差 1 nat 以上)
        if max_mid - final > 1.0:
            signal_appeared_but_lost += 1
            s["lost_signal"] = True

    pct = 100 * signal_appeared_but_lost / len(incorrect) if incorrect else 0
    print(f"  错误样本中, 中间层信号高于最终层 (>1 nat): "
          f"{signal_appeared_but_lost}/{len(incorrect)} ({pct:.1f}%)")

    # 更严格的定义: 中间层信号 > -2.3 (prob > 0.1) 但最终层 < -2.3
    strong_mid_weak_final = 0
    for s in incorrect:
        logprobs = s["correct_token_logprob"]
        if len(logprobs) < 2:
            continue
        max_mid = max(logprobs[:-1])
        final = logprobs[-1]
        if max_mid > np.log(0.1) and final < np.log(0.1):
            strong_mid_weak_final += 1
    pct2 = 100 * strong_mid_weak_final / len(incorrect) if incorrect else 0
    print(f"  错误样本中, 中间层 prob>0.1 但最终层 prob<0.1: "
          f"{strong_mid_weak_final}/{len(incorrect)} ({pct2:.1f}%)")

    # rank 统计
    print("\n--- Rank 统计 ---")
    for label, group in [("Correct", correct), ("Incorrect", incorrect)]:
        if not group:
            continue
        mid_ranks = []
        final_ranks = []
        for s in group:
            ranks = s["correct_token_rank"]
            if len(ranks) < 2:
                continue
            mid_ranks.append(min(ranks[:-1]))  # 中间层最高 rank
            final_ranks.append(ranks[-1])
        if mid_ranks:
            print(f"  {label}: 中间层最佳 rank 中位数 = {np.median(mid_ranks):.0f}, "
                  f"最终层 rank 中位数 = {np.median(final_ranks):.0f}")

    print("\n" + "=" * 60)
    print("KEY FINDING:")
    if pct > 50:
        print(f"  ✓ 在 {pct:.1f}% 的错误样本中, 正确答案信号在中间层出现过但最终丢失")
        print("  → '知道但没说' 现象存在, 值得继续扩展实验")
    elif pct > 20:
        print(f"  ? {pct:.1f}% 的错误样本有信号丢失, 处于临界区")
        print("  → 建议扩大样本量, 调整阈值")
    else:
        print(f"  ✗ 仅 {pct:.1f}% 的错误样本有信号丢失")
        print("  → '信号丢失' 现象不显著, 可能需要重新审视假设")
    print("=" * 60)
