"""
轻量脚本: 仅重跑 prediction + negatives 分析 (不重新采集轨迹).
依赖: outputs/data/full_results_Qwen3-8B.json (已存在)
"""
import json
import os
import time

import config
from analyze import (print_summary, plot_trajectory_comparison,
                       plot_pattern_distribution, compute_trajectory_metrics)
from prediction import run_prediction_task
from negatives import run_negative_controls


def main():
    print("=" * 70)
    print("Analysis Fix: Prediction + Negative Controls (using saved trajectories)")
    print("=" * 70)

    results_file = os.path.join(config.DATA_DIR, f"full_results_{config.MODEL_NAME}.json")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found. Run Step 1 first.")
        return

    print(f"\n[1/4] Loading saved trajectories from {results_file} ...")
    with open(results_file) as f:
        all_results = json.load(f)
    print(f"  Loaded {len(all_results)} samples")

    print("\n[2/4] Computing metrics...")
    for s in all_results:
        s["metrics"] = compute_trajectory_metrics(s)

    print("\n[3/4] Prediction task (with train/test split)...")
    t0 = time.time()
    pred = run_prediction_task(all_results)
    pred_file = os.path.join(config.DATA_DIR, f"prediction_results_{config.MODEL_NAME}.json")
    with open(pred_file, "w") as f:
        json.dump(pred, f, indent=2)
    print(f"  Saved: {pred_file} ({time.time()-t0:.0f}s)")

    print("\n[4/4] Negative controls (n=50 + t-test)...")
    t0 = time.time()
    from trajectory_collector import TrajectoryCollector
    from data_loader import load_all_datasets, prepare_samples
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE,
    )
    model.eval()
    collector = TrajectoryCollector(model, tokenizer)
    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)

    neg = run_negative_controls(prepared, all_results, collector)
    neg_file = os.path.join(config.DATA_DIR, f"negative_controls_{config.MODEL_NAME}.json")
    with open(neg_file, "w") as f:
        json.dump(neg, f, indent=2)
    print(f"  Saved: {neg_file} ({time.time()-t0:.0f}s)")

    print_summary(all_results)

    print(f"\n{'='*70}")
    print("DONE: Analysis fix complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()