"""
Step 1: 前向采集 + 预测 + 负对照 (第一~四章)
跑 1000 题, 不含因果干预. 代码已验证, 可直接上正式规模.
"""
import json
import os
import time
import numpy as np
import torch
from tqdm import tqdm

import config
from data_loader import load_all_datasets, prepare_samples
from trajectory_collector import TrajectoryCollector
from analyze import (print_summary, plot_trajectory_comparison,
                       plot_single_trajectory, plot_pattern_distribution,
                       compute_trajectory_metrics)
from prediction import run_prediction_task
from negatives import run_negative_controls


def is_answer_correct(generated: str, aliases: list) -> bool:
    gen_lower = generated.lower().strip()
    for ans in aliases:
        if ans.lower().strip() in gen_lower:
            return True
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
    print("Step 1: Large-scale Trajectory Collection (Chapters 1-4)")
    print(f"Model: {config.MODEL_NAME}")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE,
    )
    model.eval()

    print("\n[1/5] Loading data...")
    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    print(f"  {len(prepared)} samples")

    print("\n[2/5] Collecting trajectories...")
    collector = TrajectoryCollector(model, tokenizer)
    all_results = []
    sanity_results = []
    t0 = time.time()
    for i, s in enumerate(tqdm(prepared, desc="Collecting")):
        try:
            gen = collector.generate_answer(s["prompt_ids"])
            correct = is_answer_correct(gen["text"], s["aliases"])
            traj = collector.collect_trajectory(
                s["prompt_ids"], s["answer_token_ids"], gen["token_ids"])
            sanity_passed = (traj["final_argmax_token"] == gen["first_token_id"])
            sanity_results.append({"id": s["id"], "passed": sanity_passed})
            all_results.append({
                "id": s["id"], "task": s["task"], "question": s["question"],
                "answer": s["answer"], "aliases": s["aliases"],
                "generated": gen["text"], "final_correct": correct,
                "num_layers": traj["num_layers"],
                "correct_logprob": traj["correct_logprob"],
                "correct_rank": traj["correct_rank"],
                "generated_logprob": traj["generated_logprob"],
                "generated_rank": traj["generated_rank"],
                "cis": traj["cis"], "top5": traj["top5"],
            })
            if (i+1) % 100 == 0:
                el = time.time()-t0
                nc = sum(r["final_correct"] for r in all_results)
                ns = sum(r["passed"] for r in sanity_results)
                print(f"\n  [{i+1}/{len(prepared)}] {el:.0f}s, correct:{nc}/{len(all_results)}, sanity:{ns}/{len(sanity_results)}")
        except Exception as e:
            print(f"\n  Error {s['id']}: {e}")
    print(f"\n  Done in {time.time()-t0:.0f}s")

    print("\n[3/5] Saving data...")
    data_file = os.path.join(config.DATA_DIR, f"full_results_{config.MODEL_NAME}.json")
    with open(data_file, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  {data_file}")

    print("\n[4/5] Analysis + prediction...")
    for s in all_results:
        s["metrics"] = compute_trajectory_metrics(s)
    plot_trajectory_comparison(all_results,
        os.path.join(config.FIGURE_DIR, f"trajectory_comparison_{config.MODEL_NAME}.png"))
    plot_pattern_distribution(all_results,
        os.path.join(config.FIGURE_DIR, f"pattern_distribution_{config.MODEL_NAME}.png"))
    print_summary(all_results)
    pred = run_prediction_task(all_results)
    with open(os.path.join(config.DATA_DIR, f"prediction_results_{config.MODEL_NAME}.json"), "w") as f:
        json.dump(pred, f, indent=2)

    print("\n[5/5] Negative controls...")
    neg = run_negative_controls(prepared, all_results, collector)
    with open(os.path.join(config.DATA_DIR, f"negative_controls_{config.MODEL_NAME}.json"), "w") as f:
        json.dump(neg, f, indent=2)

    n_pass = sum(r["passed"] for r in sanity_results)
    print(f"\n{'='*70}")
    print(f"DONE: {len(all_results)} samples, sanity {n_pass}/{len(sanity_results)}")
    print(f"Correct: {sum(s['final_correct'] for s in all_results)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
