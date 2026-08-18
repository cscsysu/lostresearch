"""
Effect-matched control for gold-direction patching (reviewer round 7).

Question: is patch recovery explained by "the patch simply raises the gold
logit"? If a NON-gold perturbation that produces the SAME instantaneous
increase in the final gold logit recovers far fewer examples, then recovery
is not reducible to the logit bump.

Design (per preservation failure, n=150):
  1. Baseline forward pass: final gold logit z0, gold rank.
  2. Gold patch (same protocol as run_activation_patching, beta=4): z1,
     delta = z1 - z0, net recovery, total perturbation norm.
  3. Control A (effect-matched random direction): direction d random and
     orthogonalized against the gold unembedding; bisect scale s so the
     patched gold-logit increase matches delta. Record the norm ratio
     ||s*d|| / ||gold patch|| (direction specificity) and net recovery.
  4. Control B (effect-matched non-gold token): direction = unembedding of a
     random frequent token (top-1k by unembedding norm as frequency proxy),
     orthogonalized against gold; same bisection.

Report: net recovery (baseline-not-top1 -> top1) for gold vs A vs B, mean
rank change, and the median norm ratio required for matching.

Usage:
  python run_effect_matched_control.py --n 150 --beta 4.0
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


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
    parser.add_argument("--tol", type=float, default=0.15,
                        help="relative tolerance on the matched logit gain")
    parser.add_argument("--scale-cap", type=float, default=400.0,
                        help="max control scale (absolute) before giving up")
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
        s["is_pres"] = s["best_mid_rank"] <= args.k
    pres = [s for s in errors if s["is_pres"]]
    print(f"Preservation errors: {len(pres)}")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    unembed = model.lm_head.weight.float()
    device = model.device

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    prep_map = {s["id"]: s for s in prepared}

    rng = np.random.default_rng(42)
    # frequent-token pool: large unembedding norm as a frequency proxy
    norms = unembed.norm(dim=1)
    freq_pool = torch.topk(norms, 1000).indices.tolist()

    def run_pass(sample, peak_layer, mode, direction=None, scale=0.0):
        """One forward pass. mode: 'base' | 'gold' | 'control'.
        Returns (gold_rank, gold_logit, total_added_norm)."""
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long,
                                 device=device)
        gold_token = sample["primary_answer_ids"][0]
        w = unembed[gold_token].to(device)
        w = w / (w.norm() + 1e-6)

        captured = {}
        total_norm = 0.0

        def capture_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["peak"] = h[0, -1, :].detach().clone().float()
            return out

        def make_patch_hook(mode_, w_, d_, s_):
            def hook(module, inp, out):
                nonlocal total_norm
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                cur = h[0, -1, :].float()
                if mode_ == "gold":
                    comp = torch.clamp(captured["peak"] - cur @ w_, min=0.0)
                    add = args.beta * comp * w_
                else:
                    add = s_ * d_
                total_norm += float(add.norm())
                h[0, -1, :] = (cur + add).to(h.dtype)
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h
            return hook

        handles = [layers[peak_layer].register_forward_hook(capture_hook)]
        if mode != "base":
            for l in range(peak_layer + 1, num_layers):
                handles.append(layers[l].register_forward_hook(
                    make_patch_hook(mode, w, direction, scale)))

        with torch.no_grad():
            logits = model(input_ids, use_cache=False).logits[0, -1, :].float()
        for h in handles:
            h.remove()

        rank = int((logits > logits[gold_token]).sum().item())
        return rank, float(logits[gold_token].item()), total_norm

    eval_set = [s for s in pres if s["id"] in prep_map][:args.n]
    print(f"Evaluating {len(eval_set)} preservation failures")

    recs = []
    for meta in tqdm(eval_set, desc="Effect-matched control"):
        s = prep_map[meta["id"]]
        gold_token = s["primary_answer_ids"][0]
        w = unembed[gold_token].to(device)
        w = w / (w.norm() + 1e-6)

        r0, z0, _ = run_pass(s, meta["peak_layer"], "base")
        r1, z1, n_gold = run_pass(s, meta["peak_layer"], "gold")
        delta = z1 - z0

        rec = {"id": meta["id"], "base_rank": r0, "gold_rank": r1,
               "delta": delta, "gold_norm": n_gold}

        if delta > 0.05:  # worth matching
            for cname in ["rand", "freq"]:
                if cname == "rand":
                    t = int(rng.integers(0, unembed.shape[0]))
                else:
                    t = int(rng.choice(freq_pool))
                if t == gold_token:
                    t = int(rng.choice(freq_pool))
                d = unembed[t].to(device)
                d = d / (d.norm() + 1e-6)
                d = d - (d @ w) * w
                d = d / (d.norm() + 1e-6)

                # bisection on scale to match delta
                lo, hi = 0.0, args.scale_cap
                # check whether the cap can exceed delta at all
                _, z_hi, n_hi = run_pass(s, meta["peak_layer"], "control",
                                         d, hi)
                if z_hi - z0 < delta:
                    # cannot match even at cap: record closest
                    s_best, z_best, n_best = hi, z_hi, n_hi
                    matched = False
                else:
                    matched = True
                    for _ in range(10):
                        mid = 0.5 * (lo + hi)
                        _, z_mid, n_mid = run_pass(
                            s, meta["peak_layer"], "control", d, mid)
                        if z_mid - z0 < delta * (1 - args.tol):
                            lo = mid
                        elif z_mid - z0 > delta * (1 + args.tol):
                            hi = mid
                        else:
                            break
                    s_best = 0.5 * (lo + hi)
                    _, z_best, n_best = run_pass(s, meta["peak_layer"],
                                                 "control", d, s_best)
                # rerun control at s_best to get its rank
                r_c2, _, n_c2 = run_pass(s, meta["peak_layer"], "control",
                                         d, s_best)
                rec[cname] = {"scale": s_best, "norm": n_c2,
                              "rank": r_c2, "logit_gain": z_best - z0,
                              "matched": matched,
                              "norm_ratio": n_c2 / max(n_gold, 1e-9)}
        recs.append(rec)

    # summarize
    elig = [r for r in recs if r["base_rank"] != 0]
    print("\n" + "=" * 70)
    print(f"Effect-matched control (n={len(recs)}, eligible={len(elig)})")
    print("=" * 70)
    gold_net = sum(1 for r in elig if r["gold_rank"] == 0)
    print(f"Gold patch net recovery: {gold_net}/{len(elig)} "
          f"({100*gold_net/max(len(elig),1):.1f}%)")
    for cname in ["rand", "freq"]:
        sub = [r for r in elig if cname in r]
        if not sub:
            continue
        net = sum(1 for r in sub if r[cname]["rank"] == 0)
        ratios = [r[cname]["norm_ratio"] for r in sub]
        matched_frac = np.mean([r[cname]["matched"] for r in sub])
        print(f"{cname} control (effect-matched): net {net}/{len(sub)} "
              f"({100*net/len(sub):.1f}%), matched within tol: "
              f"{100*matched_frac:.0f}%, median norm ratio "
              f"{np.median(ratios):.1f}x")

    out_file = os.path.join(config.DATA_DIR,
                            "effect_matched_control_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"beta": args.beta, "records": recs}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
