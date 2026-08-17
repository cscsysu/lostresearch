"""
Matched-rank intervention: compare preservation vs formation recovery
AFTER matching on final gold rank.

Addresses reviewer concern: "preservation recovers more only because
its final gold rank is already close to top-1, not because of a genuine
internal mechanism difference."

Protocol:
1. For each preservation failure sample, find a formation failure sample
   with similar final gold rank (within ±2).
2. Apply same logit bonus to both matched samples.
3. Compare recovery rates -- if preservation STILL recovers more even
   after rank-matching, it's not just a rank artifact.

Usage:
  python run_matched_rank_intervention.py --n 100 --bonus 3.0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_all_datasets, prepare_samples
from prediction import extract_features
from run_cross_model import is_answer_correct


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Number of matched pairs to evaluate")
    parser.add_argument("--bonus", type=float, default=3.0)
    parser.add_argument("--rank-tolerance", type=int, default=3,
                        help="Max rank difference for matching")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LogitsProcessor, LogitsProcessorList

    # Load existing results to classify samples
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    with open(results_file) as f:
        all_results = json.load(f)

    # Classify errors into preservation / formation with final rank
    errors = [s for s in all_results if not s.get("final_correct")]
    for s in errors:
        ranks = s.get("correct_rank", [])
        s["best_mid_rank"] = min(ranks[:-1]) if len(ranks) > 1 else 1e9
        s["final_rank"] = ranks[-1] if ranks else 1e9
        s["is_preservation"] = s["best_mid_rank"] <= config.RANK_COMPETITIVE

    pres = [s for s in errors if s["is_preservation"]]
    form = [s for s in errors if not s["is_preservation"]]
    print(f"Preservation: {len(pres)}, Formation: {len(form)}")

    # Match pairs by EXACT final rank (tolerance 0 or 1)
    print(f"\nMatching pairs (tolerance ±{args.rank_tolerance})...")
    matched_pairs = []
    used_form = set()
    # Sort preservation by final rank for better matching
    pres_sorted = sorted(pres, key=lambda s: s["final_rank"])
    for p in pres_sorted:
        for i, f in enumerate(form):
            if i in used_form:
                continue
            if abs(p["final_rank"] - f["final_rank"]) <= args.rank_tolerance:
                matched_pairs.append((p, f))
                used_form.add(i)
                break
    print(f"  Matched pairs: {len(matched_pairs)}")
    if len(matched_pairs) < 10:
        print("  Too few matched pairs. Try increasing rank_tolerance.")
        return

    # Verify matching quality
    pres_ranks = [p["final_rank"] for p, _ in matched_pairs]
    form_ranks = [f["final_rank"] for _, f in matched_pairs]
    print(f"  Preservation final rank: median={np.median(pres_ranks):.0f}, mean={np.mean(pres_ranks):.1f}")
    print(f"  Formation final rank: median={np.median(form_ranks):.0f}, mean={np.mean(form_ranks):.1f}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    # Prepare samples
    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    prep_map = {s["id"]: s for s in prepared}

    class AddBonusProcessor(LogitsProcessor):
        def __init__(self, token_id, bonus):
            self.token_id = token_id
            self.bonus = bonus

        def __call__(self, input_ids, scores):
            scores = scores.clone()
            scores[:, self.token_id] = scores[:, self.token_id] + self.bonus
            return scores

    # Run intervention on matched pairs
    print(f"\nRunning matched intervention (bonus={args.bonus})...")
    pres_recovered = 0
    form_recovered = 0
    n_eval = min(args.n, len(matched_pairs))

    for p_sample, f_sample in tqdm(matched_pairs[:n_eval], desc="Matched pairs"):
        for sample, is_pres in [(p_sample, True), (f_sample, False)]:
            sid = sample["id"]
            if sid not in prep_map:
                continue
            s = prep_map[sid]
            gold_token = s["primary_answer_ids"][0]
            aliases = s["aliases"]
            input_ids = torch.tensor([s["prompt_ids"]], dtype=torch.long,
                                     device=model.device)

            procs = LogitsProcessorList([AddBonusProcessor(gold_token, args.bonus)])
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                 logits_processor=procs,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][input_ids.shape[1]:],
                                   skip_special_tokens=True).strip()
            recovered = is_answer_correct(gen, aliases)

            if is_pres:
                pres_recovered += int(recovered)
            else:
                form_recovered += int(recovered)

    # Report
    print("\n" + "=" * 70)
    print(f"Matched-Rank Intervention (n={n_eval} pairs, bonus={args.bonus})")
    print(f"  Rank tolerance: ±{args.rank_tolerance}")
    print("=" * 70)
    print(f"\n  Preservation (rank-matched): {pres_recovered}/{n_eval} ({100*pres_recovered/n_eval:.1f}%)")
    print(f"  Formation (rank-matched):    {form_recovered}/{n_eval} ({100*form_recovered/n_eval:.1f}%)")
    if form_recovered > 0:
        print(f"  Ratio: {pres_recovered/form_recovered:.1f}x")
    else:
        print(f"  Ratio: preservation >> formation (formation=0)")

    out_file = os.path.join(config.DATA_DIR, "matched_rank_intervention_Qwen3-8B.json")
    with open(out_file, "w") as f_out:
        json.dump({
            "n_pairs": n_eval, "bonus": args.bonus,
            "rank_tolerance": args.rank_tolerance,
            "pres_recovered": pres_recovered, "form_recovered": form_recovered,
            "pres_final_ranks": pres_ranks[:n_eval],
            "form_final_ranks": form_ranks[:n_eval],
        }, f_out, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
