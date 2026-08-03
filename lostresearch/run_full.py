"""
InfoDyn Full Experiment: Qwen3-8B 全流程实验.

一键跑完:
1. 数据加载 (TriviaQA 100 + HotpotQA 50 + GSM8K 50)
2. 轨迹采集 (每层 hidden state + CIS + generated 对照)
3. Sanity check
4. 信号丢失分析 + 轨迹分类
5. 预测任务 (5 个基线)
6. 负对照实验 (3 个)
7. 可视化 + 保存结果
"""
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import config
from data_loader import load_all_datasets, prepare_samples
from trajectory_collector import TrajectoryCollector
from analyze import (print_summary, plot_trajectory_comparison,
                       plot_single_trajectory, plot_pattern_distribution,
                       compute_trajectory_metrics, classify_sample)
from prediction import run_prediction_task
from negatives import run_negative_controls


def is_answer_correct(generated: str, aliases: list) -> bool:
    """检查生成的答案是否正确."""
    gen_lower = generated.lower().strip()
    for ans in aliases:
        if ans.lower().strip() in gen_lower:
            return True
    # 数字答案特殊处理
    try:
        gen_num = float(gen_lower.replace(",", "").replace("%", ""))
        for ans in aliases:
            ans_num = float(ans.lower().strip().replace(",", "").replace("%", ""))
            if abs(gen_num - ans_num) < 0.01 * max(abs(ans_num), 1):
                return True
    except (ValueError, ZeroDivisionError):
        pass
    return False


def main():
    print("=" * 70)
    print("InfoDyn Full Experiment")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Thinking: {config.ENABLE_THINKING}")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1/7] Loading model and tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE,
    )
    model.eval()
    print(f"  Model loaded on {config.DEVICE}")

    # 2. 加载数据
    print("\n[2/7] Loading data...")
    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    print(f"  Prepared {len(prepared)} samples")

    # 3. 采集轨迹
    print("\n[3/7] Collecting trajectories...")
    collector = TrajectoryCollector(model, tokenizer)
    print(f"  Layers: {collector.num_layers}")

    all_results = []
    sanity_results = []
    t0 = time.time()
    for i, sample in enumerate(tqdm(prepared, desc="Collecting")):
        try:
            # 3.1 先生成答案 (需要知道 generated token 才能算 CIS)
            gen = collector.generate_answer(sample["prompt_ids"])
            generated = gen["text"]
            correct = is_answer_correct(generated, sample["aliases"])

            # 3.2 采集轨迹 (传入 generated token 作为对照)
            traj = collector.collect_trajectory(
                prompt_ids=sample["prompt_ids"],
                answer_token_ids=sample["answer_token_ids"],
                generated_token_ids=gen["token_ids"],
            )

            # 3.3 Sanity check
            sanity_passed = (traj["final_argmax_token"] == gen["first_token_id"])

            sanity_results.append({
                "id": sample["id"],
                "question": sample["question"][:50],
                "answer": sample["answer"],
                "generated": generated[:30],
                "correct": correct,
                "final_argmax": traj["final_argmax_token"],
                "generated_first": gen["first_token_id"],
                "passed": sanity_passed,
            })

            result = {
                "id": sample["id"],
                "task": sample["task"],
                "question": sample["question"],
                "answer": sample["answer"],
                "aliases": sample["aliases"],
                "generated": generated,
                "generated_token_ids": gen["token_ids"],
                "final_correct": correct,
                "num_layers": traj["num_layers"],
                "correct_logprob": traj["correct_logprob"],
                "correct_rank": traj["correct_rank"],
                "generated_logprob": traj["generated_logprob"],
                "generated_rank": traj["generated_rank"],
                "cis": traj["cis"],
                "top5": traj["top5"],
            }
            all_results.append(result)

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                n_pass = sum(s["passed"] for s in sanity_results)
                n_correct = sum(r["final_correct"] for r in all_results)
                print(f"\n  [{i+1}/{len(prepared)}] {elapsed:.0f}s, "
                      f"correct: {n_correct}/{len(all_results)}, "
                      f"sanity: {n_pass}/{len(sanity_results)}")

        except Exception as e:
            print(f"\n  Error on {sample['id']}: {e}")
            continue

    print(f"\n  Collected {len(all_results)} trajectories in {time.time()-t0:.0f}s")

    # 4. Sanity check 报告
    print("\n[4/7] Sanity check report...")
    n_pass = sum(s["passed"] for s in sanity_results)
    n_total = len(sanity_results)
    print(f"  Passed: {n_pass}/{n_total} ({100*n_pass/max(n_total,1):.1f}%)")
    if n_pass < n_total:
        print(f"  ⚠ {n_total - n_pass} samples failed! 检查 token 对齐问题。")

    # 5. 保存数据
    print("\n[5/7] Saving data...")
    data_file = os.path.join(config.DATA_DIR, f"full_results_{config.MODEL_NAME}.json")
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  Results: {data_file}")

    sanity_file = os.path.join(config.DATA_DIR, f"sanity_{config.MODEL_NAME}.json")
    with open(sanity_file, "w", encoding="utf-8") as f:
        json.dump(sanity_results, f, ensure_ascii=False, indent=2)

    # 6. 分析 + 可视化
    print("\n[6/7] Analysis and visualization...")
    # 先计算每个样本的 metrics
    for s in all_results:
        s["metrics"] = compute_trajectory_metrics(s)

    # 总体对比图
    plot_trajectory_comparison(
        all_results,
        os.path.join(config.FIGURE_DIR, f"trajectory_comparison_{config.MODEL_NAME}.png"))

    # Pattern 分布
    plot_pattern_distribution(
        all_results,
        os.path.join(config.FIGURE_DIR, f"pattern_distribution_{config.MODEL_NAME}.png"))

    # 挑几个有意思的样本单独画
    interesting = []
    # 找一个 "Early-Decay" 的错误样本
    for s in all_results:
        if not s["final_correct"] and s["metrics"]["pattern"] == "Early-Decay":
            interesting.append(("early_decay", s))
            break
    # 找一个 "Fluctuating" 的错误样本
    for s in all_results:
        if not s["final_correct"] and s["metrics"]["pattern"] == "Fluctuating":
            interesting.append(("fluctuating", s))
            break
    # 找一个正确样本
    for s in all_results:
        if s["final_correct"]:
            interesting.append(("correct", s))
            break
    # 找一个 "Absent" 的错误样本 (真的不知道)
    for s in all_results:
        if not s["final_correct"] and s["metrics"]["pattern"] == "Absent":
            interesting.append(("absent", s))
            break

    for label, s in interesting:
        plot_single_trajectory(
            s,
            os.path.join(config.FIGURE_DIR, f"single_{label}_{s['id']}.png"))

    # 打印统计摘要
    print_summary(all_results)

    # 7. 预测任务
    print("\n[7/7] Prediction task...")
    prediction_results = run_prediction_task(all_results)

    pred_file = os.path.join(config.DATA_DIR, f"prediction_results_{config.MODEL_NAME}.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(prediction_results, f, ensure_ascii=False, indent=2)

    # 负对照
    print("\n" + "=" * 70)
    print("Bonus: Negative Controls")
    print("=" * 70)
    neg_results = run_negative_controls(prepared, all_results, collector)
    neg_file = os.path.join(config.DATA_DIR, f"negative_controls_{config.MODEL_NAME}.json")
    with open(neg_file, "w", encoding="utf-8") as f:
        json.dump(neg_results, f, ensure_ascii=False, indent=2)

    # 8. 因果干预 (第五章)
    print("\n[8/9] Causal intervention (Chapter 5)...")
    from intervention import run_intervention_experiment
    intervention_results = run_intervention_experiment(model, tokenizer, prepared, all_results)
    intv_file = os.path.join(config.DATA_DIR, f"intervention_{config.MODEL_NAME}.json")
    # 自定义序列化 (含 tensor)
    def serialize(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj
    with open(intv_file, "w", encoding="utf-8") as f:
        json.dump(serialize(intervention_results), f, ensure_ascii=False, indent=2)
    print(f"  Intervention: {intv_file}")

    # 9. Benchmark 发布 (第六章)
    print("\n[9/9] Benchmark release (Chapter 6)...")
    from infodyn_bench import create_benchmark_release
    create_benchmark_release()

    # 最终总结
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE (All 6 chapters)")
    print("=" * 70)
    print(f"Total samples: {len(all_results)}")
    print(f"Correct: {sum(s['final_correct'] for s in all_results)}")
    print(f"Incorrect: {sum(not s['final_correct'] for s in all_results)}")
    print(f"\nOutputs:")
    print(f"  Data: {config.DATA_DIR}")
    print(f"  Figures: {config.FIGURE_DIR}")
    print(f"\nKey files:")
    print(f"  - full_results_{config.MODEL_NAME}.json (轨迹数据)")
    print(f"  - prediction_results_{config.MODEL_NAME}.json (预测任务)")
    print(f"  - negative_controls_{config.MODEL_NAME}.json (负对照)")
    print(f"  - intervention_{config.MODEL_NAME}.json (因果干预)")
    print(f"  - infodyn_bench_release.json (Benchmark 发布)")
    print(f"  - trajectory_comparison_{config.MODEL_NAME}.png (总览图)")
    print(f"  - pattern_distribution_{config.MODEL_NAME}.png (模式分布)")
    print("=" * 70)


if __name__ == "__main__":
    main()
