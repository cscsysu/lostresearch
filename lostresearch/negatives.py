"""
Negative controls: 验证信号不是 probe artifact.

负对照:
1. 随机标签: 把答案换成同长度随机 token
2. 答案置换: A 题的答案配给 B 题
3. 无关位置: 取非最后位置的 hidden state
4. 层序打乱: 打乱层顺序看信号是否消失
"""
import json
import os
import random
from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F

import config
from trajectory_collector import TrajectoryCollector


def run_random_label_control(prepared_samples: List[Dict], collector: TrajectoryCollector,
                              n_control: int = 20) -> Dict:
    """负对照 1: 把答案换成随机 token, 看信号是否消失.

    注意: 需要 prepared_samples (含 prompt_ids), 不是 all_results.
    """
    print("\n  [负对照 1] 随机标签 (random token as answer)")
    vocab_size = collector.unembed.shape[0]

    random_logprobs = []
    for i, s in enumerate(prepared_samples[:n_control]):
        random_token = random.randint(0, vocab_size - 1)
        traj = collector.collect_trajectory(
            prompt_ids=s["prompt_ids"],
            answer_token_ids=[[random_token]],
            generated_token_ids=[random_token],
        )
        random_logprobs.append(traj["correct_logprob"])
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n_control}")

    all_mid_max = [max(lp[1:-1]) for lp in random_logprobs if len(lp) > 2]
    return {
        "control": "random_label",
        "n": len(random_logprobs),
        "mid_max_logprob_mean": float(np.mean(all_mid_max)) if all_mid_max else -1,
        "mid_max_logprob_std": float(np.std(all_mid_max)) if all_mid_max else 0,
    }


def run_shuffled_answer_control(prepared_samples: List[Dict], collector: TrajectoryCollector,
                                  n_control: int = 20) -> Dict:
    """负对照 2: 把 A 题的答案配给 B 题."""
    print("\n  [负对照 2] 答案置换 (A 题答案配给 B 题)")

    shuffled = prepared_samples[:n_control].copy()
    random.seed(42)
    random.shuffle(shuffled)

    shuffled_logprobs = []
    for i, (orig, shuf) in enumerate(zip(prepared_samples[:n_control], shuffled)):
        if orig["prompt_ids"] and shuf["answer_token_ids"]:
            traj = collector.collect_trajectory(
                prompt_ids=orig["prompt_ids"],
                answer_token_ids=shuf["answer_token_ids"],
                generated_token_ids=shuf.get("generated_token_ids", []),
            )
            shuffled_logprobs.append(traj["correct_logprob"])
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n_control}")

    all_mid_max = [max(lp[1:-1]) for lp in shuffled_logprobs if len(lp) > 2]
    return {
        "control": "shuffled_answer",
        "n": len(shuffled_logprobs),
        "mid_max_logprob_mean": float(np.mean(all_mid_max)) if all_mid_max else -1,
        "mid_max_logprob_std": float(np.std(all_mid_max)) if all_mid_max else 0,
    }


def run_shuffled_layer_control(samples: List[Dict], n_control: int = 20) -> Dict:
    """负对照 3: 打乱层顺序, 看信号规律是否消失."""
    print("\n  [负对照 3] 层序打乱 (shuffle layer order)")

    deltas = []
    for s in samples[:n_control]:
        logprobs = s["correct_logprob"]
        if len(logprobs) < 4:
            continue
        # 打乱层顺序
        shuffled = logprobs.copy()
        random.shuffle(shuffled)
        # 重新计算 mid-final delta
        if len(shuffled) > 2:
            delta = max(shuffled[1:-1]) - shuffled[-1]
            deltas.append(delta)

    return {
        "control": "shuffled_layer",
        "n": len(deltas),
        "mid_final_delta_mean": float(np.mean(deltas)) if deltas else 0,
        "mid_final_delta_std": float(np.std(deltas)) if deltas else 0,
    }


def run_negative_controls(prepared_samples: List[Dict], all_results: List[Dict],
                           collector: TrajectoryCollector) -> List[Dict]:
    """运行所有负对照.

    Args:
        prepared_samples: 含 prompt_ids 的原始样本 (用于跑 forward)
        all_results: 已采集的结果 (用于对比)
        collector: TrajectoryCollector
    """
    print("\n" + "=" * 70)
    print("NEGATIVE CONTROLS")
    print("=" * 70)

    results = []
    n_control = min(20, len(prepared_samples))

    # 1. 随机标签
    try:
        r1 = run_random_label_control(prepared_samples, collector, n_control)
        results.append(r1)
        print(f"  随机标签 mid_max_logprob: {r1['mid_max_logprob_mean']:.2f} "
              f"± {r1['mid_max_logprob_std']:.2f}")
    except Exception as e:
        print(f"  随机标签失败: {e}")

    # 2. 答案置换
    try:
        r2 = run_shuffled_answer_control(prepared_samples, collector, n_control)
        results.append(r2)
        print(f"  答案置换 mid_max_logprob: {r2['mid_max_logprob_mean']:.2f} "
              f"± {r2['mid_max_logprob_std']:.2f}")
    except Exception as e:
        print(f"  答案置换失败: {e}")

    # 3. 层序打乱 (用 all_results 的已有轨迹)
    try:
        r3 = run_shuffled_layer_control(all_results, n_control)
        results.append(r3)
        print(f"  层序打乱 mid_final_delta: {r3['mid_final_delta_mean']:.2f} "
              f"± {r3['mid_final_delta_std']:.2f}")
    except Exception as e:
        print(f"  层序打乱失败: {e}")

    # 对比: 真实样本的信号
    real_mid_max = [max(s["correct_logprob"][1:-1])
                    for s in all_results if len(s["correct_logprob"]) > 2]
    print(f"\n  真实样本 mid_max_logprob: {np.mean(real_mid_max):.2f} "
          f"± {np.std(real_mid_max):.2f}")
    print("\n  判据: 负对照的信号应显著弱于真实样本")
    for r in results:
        if r["control"] in ["random_label", "shuffled_answer"]:
            if r["mid_max_logprob_mean"] < np.mean(real_mid_max) - 2 * np.std(real_mid_max):
                print(f"  ✓ {r['control']} 信号显著弱于真实 (通过)")
            else:
                print(f"  ? {r['control']} 信号未显著弱于真实 (需检查)")

    return results
