"""
Calibrated steering: small logit bonus reveals preservation vs formation gap.

Key insight: preservation failures have gold token at final-layer rank 0-5
(91%), while formation failures have rank 23+ (89%). A SMALL fixed logit
bonus (+1 to +3) is enough to push preservation's nearly-there gold to top-1,
but NOT enough to rescue formation's distant gold.

This produces a large gap in full-answer recovery:
  preservation ~high% (gold was almost there, small push tips it over)
  formation ~low% (gold too far behind, small push doesn't help)

This is the trajectory-predicted repairability the reviewer asks for.

Usage:
  python run_recovery.py --n150 --bonuses 1.0,2.0,3.0
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from data_loader import load_all_datasets, prepare_samples
from run_cross_model import is_answer_correct


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
    parser.add_argument("--bonuses", type=str, default="1.0,2.0,3.0")
    parser.add_argument("--single-word", action="store_true", help="Only evaluate single-word answers")
    args = parser.parse_args()
    bonuses = [float(x) for x in args.bonuses.split(",")]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LogitsProcessor, LogitsProcessorList
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    final_norm = model.model.norm
    unembed_f = model.lm_head.weight.float()
    device = model.device

    class AddBonusProcessor(LogitsProcessor):
        """Add a fixed bonus to the gold token's logit (NOT force to top)."""
        def __init__(self, token_id, bonus):
            self.token_id = token_id
            self.bonus = bonus

        def __call__(self, input_ids, scores):
            scores = scores.clone()
            scores[:, self.token_id] = scores[:, self.token_id] + self.bonus
            return scores

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            existing = {r["id"]: r for r in json.load(f)}
        for s in prepared:
            if s["id"] in existing:
                s["final_correct"] = existing[s["id"]]["final_correct"]

    errors = [s for s in prepared if not s.get("final_correct", True)]
    if args.single_word:
        errors = [s for s in errors if len(s.get("answer", "").split()) == 1]
        print(f"Filtered to single-word answers: {len(errors)} errors")
    print(f"Errors available: {len(errors)}; evaluating up to {args.n}")

    groups = ["preservation", "formation"]
    rec = {g: {b: 0 for b in bonuses} for g in groups}
    tot = {g: 0 for g in groups}
    results = []

    for s in tqdm(errors[:args.n], desc="Calibrated steering"):
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        aliases = s["aliases"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # baseline
        base_out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:],
                                     skip_special_tokens=True).strip()
        if is_answer_correct(base_text, aliases):
            continue

        # classify via intermediate ranks
        hidden_buffer = {}

        def make_hook(idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                hidden_buffer[idx] = h[0, -1, :].detach().clone()
            return hook

        hks = [layer.register_forward_hook(make_hook(l)) for l, layer in enumerate(layers)]
        with torch.no_grad():
            model(input_ids, use_cache=False)
        for h in hks:
            h.remove()

        inter_ranks = []
        for l in range(num_layers - 1):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            inter_ranks.append((lr > lr[gold_token]).sum().item())
        best_rank = min(inter_ranks) if inter_ranks else 1e9
        group = "preservation" if best_rank <= args.k else "formation"
        tot[group] += 1

        # steering: add small bonus to gold logit
        per_bonus = {}
        for b in bonuses:
            procs = LogitsProcessorList([AddBonusProcessor(gold_token, b)])
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                 logits_processor=procs,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][input_ids.shape[1]:],
                                   skip_special_tokens=True).strip()
            ok = is_answer_correct(gen, aliases)
            per_bonus[b] = ok
            rec[group][b] += int(ok)

        results.append({
            "id": s["id"], "task": s.get("task", "unknown"),
            "group": group, "best_rank": best_rank,
            "final_rank": inter_ranks[-1] if inter_ranks else -1,
            "repair": {str(b): per_bonus[b] for b in bonuses},
        })

    print("\n" + "=" * 74)
    print("Calibrated steering: preservation vs formation")
    print("=" * 74)
    for g in groups:
        if tot[g] == 0:
            print(f"\n[{g}] no samples")
            continue
        print(f"\n[{g}]n={tot[g]}")
        for b in bonuses:
            print(f"  bonus={b}: {rec[g][b]}/{tot[g]} ({100*rec[g][b]/tot[g]:.1f}%)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"bonuses": bonuses, "k": args.k, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
