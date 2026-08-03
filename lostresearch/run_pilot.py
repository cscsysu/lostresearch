"""Main pilot experiment: 跑 N 题, 采集每层 CIS 轨迹, 分析 '信号丢失' 现象."""
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import config
from data_loader import load_triviaqa, prepare_samples, build_prompt
from trajectory_collector import TrajectoryCollector
from visualize import plot_single_trajectory, plot_correct_vs_incorrect, print_summary


def is_answer_correct(generated: str, aliases: list) -> bool:
    """检查生成的答案是否正确 (简单字符串匹配)."""
    gen_lower = generated.lower().strip()
    for ans in aliases:
        if ans.lower().strip() in gen_lower:
            return True
    return False


def main():
    print("=" * 60)
    print("InfoDyn Pilot Experiment")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Thinking: {config.ENABLE_THINKING}")
    print(f"Samples: {config.NUM_SAMPLES}")
    print("=" * 60)

    # 1. 加载 tokenizer 和 model
    print("\n[1/5] Loading model and tokenizer...")
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
    print("\n[2/5] Loading data...")
    samples = load_triviaqa(config.NUM_SAMPLES)
    prepared = prepare_samples(samples, tokenizer)
    print(f"  Prepared {len(prepared)} samples")

    # 3. 采集轨迹
    print("\n[3/5] Collecting trajectories...")
    collector = TrajectoryCollector(model, tokenizer)
    print(f"  Number of transformer layers: {collector.num_layers}")

    all_results = []
    t0 = time.time()
    for i, sample in enumerate(tqdm(prepared, desc="Collecting")):
        try:
            # 3.1 采集轨迹 (forward pass)
            traj = collector.collect_trajectory(
                prompt_ids=sample["prompt_ids"],
                primary_answer_ids=sample["primary_answer_ids"],
            )

            # 3.2 生成答案
            generated = collector.generate_answer(sample["prompt_ids"])
            correct = is_answer_correct(generated, sample["aliases"])

            result = {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "aliases": sample["aliases"],
                "generated": generated,
                "final_correct": correct,
                "num_layers": traj["num_layers_collected"],
                "correct_token_logprob": traj["correct_token_logprob"],
                "correct_token_rank": traj["correct_token_rank"],
                "top5_per_layer": traj["top5_per_layer"],
            }
            all_results.append(result)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"\n  [{i+1}/{len(prepared)}] {elapsed:.1f}s elapsed, "
                      f"correct so far: {sum(r['final_correct'] for r in all_results)}/{len(all_results)}")

        except Exception as e:
            print(f"\n  Error on sample {sample['id']}: {e}")
            continue

    print(f"\n  Collected {len(all_results)} trajectories in {time.time()-t0:.1f}s")

    # 4. 保存结果
    print("\n[4/5] Saving results...")
    output_file = os.path.join(config.DATA_DIR, f"pilot_results_{config.MODEL_NAME}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {output_file}")

    # 5. 可视化和统计
    print("\n[5/5] Visualization and analysis...")

    # 5.1 总体对比图
    plot_correct_vs_incorrect(
        all_results,
        save_path=os.path.join(config.FIGURE_DIR, f"trajectory_comparison_{config.MODEL_NAME}.png")
    )

    # 5.2 挑几个有意思的样本单独画
    # 找一个错误样本, 一个正确样本
    correct_sample = next((s for s in all_results if s["final_correct"]), None)
    incorrect_sample = next((s for s in all_results if not s["final_correct"]), None)

    if correct_sample:
        plot_single_trajectory(
            correct_sample,
            save_path=os.path.join(config.FIGURE_DIR, f"single_correct_{correct_sample['id']}.png")
        )
    if incorrect_sample:
        plot_single_trajectory(
            incorrect_sample,
            save_path=os.path.join(config.FIGURE_DIR, f"single_incorrect_{incorrect_sample['id']}.png")
        )

    # 5.3 打印统计摘要
    print_summary(all_results)

    print(f"\nDone! Figures saved to: {config.FIGURE_DIR}")
    print(f"Data saved to: {config.DATA_DIR}")


if __name__ == "__main__":
    main()
