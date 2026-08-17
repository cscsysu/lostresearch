"""
Peak-state restoration: intervene at the PEAK layer, not the output.

Key design insight: previous matched-rank experiments failed because they
intervened at the OUTPUT logit level, which trivially depends on final rank.
This experiment instead intervenes at the layer where the gold signal peaked.

Protocol:
1. For each error, find the layer L_peak where gold rank was best.
2. Take the residual stream at L_peak (where gold was strongest).
3. Propagate that state forward, replacing the residual at all subsequent
   layers with a blend: h_l' = (1-alpha)*h_l + alpha*h_{L_peak}
   (i.e., "hold onto" the peak state instead of letting it decay)
4. Measure whether the final answer becomes correct.

Rationale:
- Preservation failures HAVE a good peak state to restore -> should recover
- Formation failures have NO good peak state -> restoration is useless
This decouples the effect from final rank, because we intervene based on
the INTERMEDIATE state, not the output.

Usage:
  python run_peak_restoration.py --n 120 --alpha 0.5
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
from run_cross_model import is_answer_correct


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend weight for peak state (0=no change, 1=full replace)")
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load classification data
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as f:
        all_results = json.load(f)

    existing = {r["id"]: r for r in all_results}
    errors_meta = [r for r in all_results if not r.get("final_correct")]
    for s in errors_meta:
        ranks = s.get("correct_rank", [])
        inter = ranks[:-1] if len(ranks) > 1 else ranks
        s["best_mid_rank"] = min(inter) if inter else 10**9
        s["peak_layer"] = int(np.argmin(inter)) if inter else 0
        s["final_rank"] = ranks[-1] if ranks else 10**9
        s["is_pres"] = s["best_mid_rank"] <= args.k

    pres_meta = [s for s in errors_meta if s["is_pres"]]
    form_meta = [s for s in errors_meta if not s["is_pres"]]
    print(f"Preservation: {len(pres_meta)}, Formation: {len(form_meta)}")
    print(f"  Preservation peak layer: median={np.median([s['peak_layer'] for s in pres_meta]):.0f}")
    print(f"  Formation peak layer: median={np.median([s['peak_layer'] for s in form_meta]):.0f}")

    # Load model
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
        """Generate while holding the residual stream near its peak-layer state.

        We capture the residual at peak_layer, then for all layers after it we
        blend the current residual with the captured peak residual. This
        simulates 'not letting the signal decay after its peak'.
        """
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

    # Balanced evaluation
    half = args.n // 2
    pres_eval = [s for s in pres_meta if s["id"] in prep_map][:half]
    form_eval = [s for s in form_meta if s["id"] in prep_map][:half]

    stats = {"preservation": {"n": 0, "rec": 0}, "formation": {"n": 0, "rec": 0}}
    records = []

    print(f"\nRunning peak restoration (alpha={args.alpha})...")
    for group, group_meta in [("preservation", pres_eval), ("formation", form_eval)]:
        for meta in tqdm(group_meta, desc=group):
            s = prep_map[meta["id"]]
            gen = generate_with_peak_restoration(s, meta["peak_layer"], args.alpha)
            rec = is_answer_correct(gen, s["aliases"])
            stats[group]["n"] += 1
            stats[group]["rec"] += int(rec)
            records.append({
                "id": meta["id"], "group": group,
                "peak_layer": meta["peak_layer"],
                "best_mid_rank": meta["best_mid_rank"],
                "final_rank": meta["final_rank"],
                "recovered": rec,
            })

    print("\n" + "=" * 70)
    print(f"Peak-State Restoration Results (alpha={args.alpha})")
    print("=" * 70)
    for group in ["preservation", "formation"]:
        st = stats[group]
        if st["n"]:
            print(f"\n[{group}] n={st['n']}")
            print(f"  Recovered: {st['rec']}/{st['n']} ({100*st['rec']/st['n']:.1f}%)")
    p_rate = stats["preservation"]["rec"] / max(stats["preservation"]["n"], 1)
    f_rate = stats["formation"]["rec"] / max(stats["formation"]["n"], 1)
    if f_rate > 0:
        print(f"\n  Ratio: {p_rate/f_rate:.1f}x")
    else:
        print(f"\n  Formation recovered 0 -> preservation-exclusive effect")

    out_file = os.path.join(config.DATA_DIR, "peak_restoration_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"alpha": args.alpha, "stats": stats, "records": records}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
