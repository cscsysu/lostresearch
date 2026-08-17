"""
Run trajectory collection ONLY for datasets not already in existing results.
Automatically detects which tasks have been completed, skips them, and merges.

Usage:
  python run_new_tasks_only.py
"""
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import (load_triviaqa, load_hotpotqa, load_gsm8k,
                         load_commonsenseqa, load_arc_challenge,
                         load_quality, load_mmlu, prepare_samples)
from trajectory_collector import TrajectoryCollector
from run_cross_model import is_answer_correct


# All available datasets and their loaders
ALL_TASKS = {
    "triviaqa": {"loader": load_triviaqa, "n": 500},
    "hotpotqa": {"loader": load_hotpotqa, "n": 250},
    "gsm8k": {"loader": load_gsm8k, "n": 250},
    "commonsenseqa": {"loader": load_commonsenseqa, "n": 250},
    "arc_challenge": {"loader": load_arc_challenge, "n": 250},
    "quality": {"loader": load_quality, "n": 200},
    "mmlu": {"loader": load_mmlu, "n": 200},
}


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 70)
    print("Smart runner: only collect trajectories for MISSING tasks")
    print("=" * 70)

    # Check existing results
    existing_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    existing_results = []
    existing_tasks = set()
    if os.path.exists(existing_file):
        with open(existing_file) as f:
            existing_results = json.load(f)
        existing_tasks = set(r.get("task") for r in existing_results)
        print(f"\nExisting results: {len(existing_results)} samples")
        print(f"  Tasks already done: {sorted(existing_tasks)}")

    # Determine which tasks to run
    tasks_to_run = {k: v for k, v in ALL_TASKS.items() if k not in existing_tasks}
    if not tasks_to_run:
        print("\nAll tasks already completed! Nothing to do.")
        return

    print(f"\n  Tasks to run: {sorted(tasks_to_run.keys())}")

    # Load model
    print("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    # Load only missing datasets
    print("\n[2] Loading missing datasets...")
    new_samples = []
    for task_name, task_cfg in tasks_to_run.items():
        print(f"  Loading {task_name} (n={task_cfg['n']}) ...")
        try:
            samples = task_cfg["loader"](task_cfg["n"])
            new_samples.extend(samples)
            print(f"    Loaded {len(samples)} samples")
        except Exception as e:
            print(f"    Failed: {e}")

    if not new_samples:
        print("\nNo new samples loaded. Check network/data availability.")
        return

    print(f"\nTotal new samples: {len(new_samples)}")

    # Prepare (tokenize)
    prepared = prepare_samples(new_samples, tokenizer)
    print(f"Prepared {len(prepared)} samples")

    # Collect trajectories
    print("\n[3] Collecting trajectories...")
    collector = TrajectoryCollector(model, tokenizer)
    results = []

    for s in tqdm(prepared, desc="New tasks"):
        try:
            # Generate answer first
            input_ids = torch.tensor([s["prompt_ids"]], dtype=torch.long, device=model.device)
            out = model.generate(input_ids, max_new_tokens=config.MAX_NEW_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            gen_ids = out[0][input_ids.shape[1]:].tolist()
            generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            correct = is_answer_correct(generated, s["aliases"])

            # Collect trajectory
            answer_token_ids = [s["primary_answer_ids"]] if "primary_answer_ids" in s else [[]]
            traj = collector.collect_trajectory(
                s["prompt_ids"],
                answer_token_ids=answer_token_ids,
                generated_token_ids=gen_ids
            )

            results.append({
                "id": s["id"],
                "task": s.get("task", "unknown"),
                "question": s["question"],
                "answer": s["answer"],
                "aliases": s["aliases"],
                "generated": generated,
                "final_correct": correct,
                "correct_logprob": traj["correct_logprob"],
                "correct_rank": traj["correct_rank"],
                "multitoken_best_rank": traj.get("multitoken_best_rank", traj["correct_rank"]),
                "cis": traj["cis"],
                "num_layers": traj["num_layers"],
            })
        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # Save new results
    new_file = os.path.join(config.DATA_DIR, "new_tasks_results_Qwen3-8B.json")
    with open(new_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"\nSaved new task results: {new_file}")
    print(f"  New: {len(results)}, Correct: {sum(1 for r in results if r['final_correct'])}, "
          f"Incorrect: {sum(1 for r in results if not r['final_correct'])}")

    # Merge with existing
    existing_ids = {r["id"] for r in existing_results}
    new_to_add = [r for r in results if r["id"] not in existing_ids]
    merged = existing_results + new_to_add

    # Save merged as the new full results
    with open(existing_file, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"\nMerged into: {existing_file}")
    print(f"  Total: {len(merged)} samples ({len(existing_results)} existing + {len(new_to_add)} new)")

    # Summary by task
    print("\n  Per-task breakdown:")
    from collections import Counter
    task_counts = Counter(r.get("task") for r in merged)
    for task, count in sorted(task_counts.items()):
        print(f"    {task}: {count}")


if __name__ == "__main__":
    main()
