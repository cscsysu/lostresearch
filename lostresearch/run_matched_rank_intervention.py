"""
Matched-rank intervention (v2): rank-matched pairs + PEAK-STATE restoration.

Reviewer concern this addresses:
  "Preservation recovers more only because its final gold rank is already
   closer to top-1, not because of a genuine internal-mechanism difference."

Why the v1 design failed (logit bonus, 1.4x, p=0.33):
  v1 matched preservation and formation on FINAL gold rank, then applied an
  OUTPUT-level logit bonus. But once you equalize final rank, you equalize
  exactly the quantity a logit bonus exploits, so the design structurally
  cannot reveal a mechanism difference. The intervention acted on the output,
  not on the (very different) intermediate states.

v2 design:
  1. Match each preservation failure to a formation failure with the SAME
     final gold rank (+-tol). This removes the "final rank is closer" confound.
  2. Instead of an output logit bonus, apply PEAK-STATE RESTORATION: hold the
     residual stream near its own peak-layer state for all later layers.
     This intervenes on the INTERMEDIATE state, which is genuinely different
     between the two groups even after final-rank matching:
       - preservation: a strong peak state exists -> restoring it helps
       - formation:    no strong peak state exists -> restoring it is useless
  If preservation STILL recovers more even after final-rank matching, the
  effect cannot be a final-rank artifact -- it reflects the intermediate
  mechanism.

Usage:
  python run_matched_rank_intervention.py --n 100 --alpha 0.8 --rank-tolerance 3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_all_datasets, prepare_samples
from run_cross_model import is_answer_correct


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Max matched pairs to evaluate")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Peak-state blend weight (0=off, 1=full replace)")
    parser.add_argument("--rank-tolerance", type=int, default=3,
                        help="Max final-rank difference for matching")
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as f:
        all_results = json.load(f)

    errors = [s for s in all_results if not s.get("final_correct")]
    for s in errors:
        ranks = s.get("correct_rank", [])
        inter = ranks[:-1] if len(ranks) > 1 else ranks
        s["best_mid_rank"] = min(inter) if inter else 10**9
        s["peak_layer"] = int(np.argmin(inter)) if inter else 0
        s["final_rank"] = ranks[-1] if ranks else 10**9
        s["is_preservation"] = s["best_mid_rank"] <= args.k

    pres = [s for s in errors if s["is_preservation"]]
    form = [s for s in errors if not s["is_preservation"]]
    print(f"Preservation: {len(pres)}, Formation: {len(form)}")

    # Match pairs by final rank (removes the "closer final rank" confound)
    print(f"\nMatching pairs on FINAL rank (tolerance +-{args.rank_tolerance})...")
    matched_pairs = []
    used_form = set()
    pres_sorted = sorted(pres, key=lambda s: s["final_rank"])
    for p in pres_sorted:
        for i, fm in enumerate(form):
            if i in used_form:
                continue
            if abs(p["final_rank"] - fm["final_rank"]) <= args.rank_tolerance:
                matched_pairs.append((p, fm))
                used_form.add(i)
                break
    print(f"  Matched pairs: {len(matched_pairs)}")
    if len(matched_pairs) < 10:
        print("  Too few matched pairs. Increase rank_tolerance.")
        return

    pres_ranks = [p["final_rank"] for p, _ in matched_pairs]
    form_ranks = [fm["final_rank"] for _, fm in matched_pairs]
    print(f"  Preservation final rank: median={np.median(pres_ranks):.0f}, mean={np.mean(pres_ranks):.1f}")
    print(f"  Formation    final rank: median={np.median(form_ranks):.0f}, mean={np.mean(form_ranks):.1f}")
    # Sanity: matched groups have (near) identical final-rank distribution
    print(f"  Peak-layer gold rank -- pres median={np.median([p['best_mid_rank'] for p,_ in matched_pairs]):.0f}, "
          f"form median={np.median([fm['best_mid_rank'] for _,fm in matched_pairs]):.0f}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    device = model.device

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    prep_map = {s["id"]: s for s in prepared}

    def generate_with_peak_restoration(sample, peak_layer, alpha):
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long, device=device)
        captured = {}

        def capture_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["peak"] = h[0, -1, :].detach().clone()
            return out

        def blend_hook(module, inp, out):
            if "peak" not in captured:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            peak_vec = captured["peak"].to(h.device, h.dtype)
            h[0, -1, :] = (1 - alpha) * h[0, -1, :] + alpha * peak_vec
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        handles = [layers[peak_layer].register_forward_hook(capture_hook)]
        for l in range(peak_layer + 1, num_layers):
            handles.append(layers[l].register_forward_hook(blend_hook))
        out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for h in handles:
            h.remove()
        return tokenizer.decode(out[0][input_ids.shape[1]:],
                                skip_special_tokens=True).strip()

    print(f"\nRunning matched-rank PEAK restoration (alpha={args.alpha})...")
    pres_recovered = 0
    form_recovered = 0
    n_eval = min(args.n, len(matched_pairs))
    records = []

    for p_meta, f_meta in tqdm(matched_pairs[:n_eval], desc="Matched pairs"):
        for meta, is_pres in [(p_meta, True), (f_meta, False)]:
            sid = meta["id"]
            if sid not in prep_map:
                continue
            s = prep_map[sid]
            gen = generate_with_peak_restoration(s, meta["peak_layer"], args.alpha)
            rec = is_answer_correct(gen, s["aliases"])
            if is_pres:
                pres_recovered += int(rec)
            else:
                form_recovered += int(rec)
            records.append({"id": sid, "is_pres": is_pres,
                            "final_rank": meta["final_rank"],
                            "peak_layer": meta["peak_layer"],
                            "recovered": rec})

    print("\n" + "=" * 70)
    print(f"Matched-Rank PEAK Restoration (n={n_eval} pairs, alpha={args.alpha})")
    print(f"  Final-rank matched within +-{args.rank_tolerance}")
    print("=" * 70)
    print(f"\n  Preservation (rank-matched): {pres_recovered}/{n_eval} ({100*pres_recovered/n_eval:.1f}%)")
    print(f"  Formation    (rank-matched): {form_recovered}/{n_eval} ({100*form_recovered/n_eval:.1f}%)")
    if form_recovered > 0:
        print(f"  Ratio: {pres_recovered/form_recovered:.1f}x")
    else:
        print(f"  Ratio: preservation-exclusive (formation=0)")

    out_file = os.path.join(config.DATA_DIR, "matched_rank_intervention_Qwen3-8B.json")
    with open(out_file, "w") as f_out:
        json.dump({
            "n_pairs": n_eval, "alpha": args.alpha,
            "rank_tolerance": args.rank_tolerance,
            "pres_recovered": pres_recovered, "form_recovered": form_recovered,
            "pres_final_ranks": pres_ranks[:n_eval],
            "form_final_ranks": form_ranks[:n_eval],
            "records": records,
        }, f_out, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
