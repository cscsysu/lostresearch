"""
Run trajectory collection ONLY for newly added datasets (QuALITY + MMLU).
Appends results to existing full_results file without re-running old tasks.

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
from data_loader import load_quality, load_mmlu, prepare_samples
from trajectory_collector import TrajectoryCollector
from run_cross_model import is_answer_correct


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 70)
    print("Running ONLY new tasks: QuALITY + MMLU")
    print("=" * 70)

    # Load model
    print("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    # Load only new datasets
    print("\n[2] Loading new datasets...")
    new_samples = []

    print("Loading quality (n=200) ...")
    try:
        quality = load_quality(200)
        new_samples.extend(quality)
        print(f"  Loaded {len(quality)} samples")
    except Exception as e:
        print(f"  Failed: {e}")

    print("Loading mmlu (n=200) ...")
    try:
        mmlu = load_mmlu(200)
        new_samples.extend(mmlu)
        print(f"  Loaded {len(mmlu)} samples")
    except Exception as e:
        print(f"  Failed: {e}")

    print(f"Total new samples: {len(new_samples)}")

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
                "cis": traj["cis"],
                "num_layers": traj["num_layers"],
            })
        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # Save new results separately
    out_file = os.path.join(config.DATA_DIR, "new_tasks_results_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"\nSaved new task results: {out_file}")
    print(f"  Total: {len(results)}, Correct: {sum(1 for r in results if r['final_correct'])}, "
          f"Incorrect: {sum(1 for r in results if not r['final_correct'])}")

    # Also try to merge with existing results
    existing_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if os.path.exists(existing_file):
        with open(existing_file) as f:
            existing = json.load(f)
        # Merge (append new, avoid duplicates)
        existing_ids = {r["id"] for r in existing}
        new_to_add = [r for r in results if r["id"] not in existing_ids]
        merged = existing + new_to_add
        merged_file = os.path.join(config.DATA_DIR, "full_results_7tasks_Qwen3-8B.json")
        with open(merged_file, "w") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2,
                      default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        print(f"  Merged with existing: {merged_file} ({len(merged)} total samples)")


if __name__ == "__main__":
    main()
