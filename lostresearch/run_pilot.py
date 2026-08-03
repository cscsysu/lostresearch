"""Main pilot experiment: 跑 N 题, 采集每层 CIS 轨迹, 分析 '信号丢失' 现象.

修正版:
1. 适配新的 trajectory_collector return 格式
2. 加 final-layer sanity check 报告
3. 默认跑 10 样本 (调试 token 对齐)
"""
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
    print("InfoDyn Pilot Experiment (v2: token-aligned)")
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
    sanity_results = []
    t0 = time.time()
    for i, sample in enumerate(tqdm(prepared, desc="Collecting")):
        try:
            # 3.1 采集轨迹 (forward pass)
            traj = collector.collect_trajectory(
                prompt_ids=sample["prompt_ids"],
                answer_token_ids=sample["answer_token_ids"],
            )

            # 3.2 生成答案
            gen = collector.generate_answer(sample["prompt_ids"])
            generated = gen["text"]
            correct = is_answer_correct(generated, sample["aliases"])

            # 3.3 Sanity check: final layer argmax 是否等于生成的第一个 token
            sanity = traj["sanity_check"]
            sanity_passed = (traj["final_argmax_token"] == gen["first_token_id"])
            sanity["passed"] = sanity_passed
            sanity["generated_first_token_id"] = gen["first_token_id"]
            sanity["generated_first_token_decoded"] = gen["first_token_decoded"]
            sanity_results.append({
                "id": sample["id"],
                "question": sample["question"][:50],
                "answer": sample["answer"],
                "generated": generated,
                "correct": correct,
                "final_argmax_token": sanity["final_argmax_token"],
                "final_argmax_decoded": sanity["final_argmax_decoded"],
                "generated_first_token_id": gen["first_token_id"],
                "generated_first_token_decoded": gen["first_token_decoded"],
                "sanity_passed": sanity_passed,
            })

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
                "best_alias_idx_per_layer": traj["best_alias_idx_per_layer"],
            }
            all_results.append(result)

            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                n_pass = sum(s["sanity_passed"] for s in sanity_results)
                print(f"\n  [{i+1}/{len(prepared)}] {elapsed:.1f}s, "
                      f"correct: {sum(r['final_correct'] for r in all_results)}/{len(all_results)}, "
                      f"sanity: {n_pass}/{len(sanity_results)}")

        except Exception as e:
            print(f"\n  Error on sample {sample['id']}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"\n  Collected {len(all_results)} trajectories in {time.time()-t0:.1f}s")

    # 4. 保存结果
    print("\n[4/5] Saving results...")
    output_file = os.path.join(config.DATA_DIR, f"pilot_results_{config.MODEL_NAME}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {output_file}")

    # 保存 sanity check 结果
    sanity_file = os.path.join(config.DATA_DIR, f"sanity_check_{config.MODEL_NAME}.json")
    with open(sanity_file, "w", encoding="utf-8") as f:
        json.dump(sanity_results, f, ensure_ascii=False, indent=2)
    print(f"  Sanity check saved to {sanity_file}")

    # 5. 打印 sanity check 报告
    print("\n" + "=" * 60)
    print("SANITY CHECK REPORT")
    print("=" * 60)
    n_pass = sum(s["sanity_passed"] for s in sanity_results)
    n_total = len(sanity_results)
    print(f"Passed: {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)")
    print()
    print(f"{'ID':<16} {'Correct':<8} {'Argmax':<20} {'Generated':<20} {'Pass'}")
    print("-" * 80)
    for s in sanity_results:
        print(f"{s['id']:<16} {'✓' if s['correct'] else '✗':<8} "
              f"{repr(s['final_argmax_decoded']):<20} "
              f"{repr(s['generated_first_token_decoded']):<20} "
              f"{'✓' if s['sanity_passed'] else '✗'}")

    if n_pass < n_total:
        print(f"\n⚠ {n_total - n_pass} samples failed sanity check!")
        print("  这说明 hook/forward 的最终层 argmax 和 generate 的首 token 不一致,")
        print("  可能是 attention mask、padding 或 hook 位置问题, 需要修复后再扩展实验。")
    else:
        print(f"\n✓ 所有样本通过 sanity check, token 对齐正确, 可以继续扩展实验。")

    # 6. 可视化和统计
    print("\n[5/5] Visualization and analysis...")
    plot_correct_vs_incorrect(
        all_results,
        save_path=os.path.join(config.FIGURE_DIR, f"trajectory_comparison_{config.MODEL_NAME}.png")
    )

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

    print_summary(all_results)

    print(f"\nDone! Figures: {config.FIGURE_DIR}")
    print(f"Data: {config.DATA_DIR}")


if __name__ == "__main__":
    main()
