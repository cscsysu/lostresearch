"""
Activation patching (v4): causal peak-state patch with a matched control.

Reviewer concern this addresses:
  "Attribution != causality. A DLA layer ranking is correlational. Does
   intervening on the identified late-layer computation actually change the
   gold signal's fate, and is the effect specific to preservation failures?"

Why the previous version failed (contrast-direction MLP ablation, 0.8x):
  Projecting the (w_gold - w_comp) direction out of a SINGLE layer's MLP
  output (i) barely moves the accumulated residual and (ii) gives formation
  failures more apparent benefit purely because they have more "rank room"
  below the gold token. It measured rank room, not the decay mechanism.

v4 design (causal patch, not ablation):
  For each error we know its peak layer L* (where gold rank was best) and its
  final gold rank. We run a forward pass and, at every layer AFTER L*, we
  PATCH the last-token residual by adding back the gold-direction component
  that the peak state carried but the later layers discarded:

     delta = h_{L*} - h_l   (what decayed between peak and layer l)
     h_l' = h_l + beta * P_gold(delta)

  where P_gold projects onto the residual-space direction that maximally
  loads the gold token (the unembedding row for the gold token, pulled back
  through the final norm). We then measure the change in final gold rank.

  Control (matched-null patch): identical procedure but projecting onto a
  RANDOM competitor token's direction instead of gold. If the gold patch
  helps preservation far more than (a) the same patch on formation and
  (b) the matched-null patch, the late-layer decay is causally carrying the
  gold signal specifically -- attribution is validated as causal.

Usage:
  python run_activation_patching.py --n 120 --beta 4.0
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
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--beta", type=float, default=4.0,
                        help="Gold-direction restoration strength")
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

    existing = {r["id"]: r for r in all_results}
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

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    unembed = model.lm_head.weight.float()  # [vocab, d]
    vocab_size = unembed.shape[0]
    device = model.device

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    prep_map = {s["id"]: s for s in prepared}

    rng = np.random.default_rng(0)

    def final_gold_rank_with_patch(sample, peak_layer, dir_vec, beta):
        """Forward pass; after peak_layer, restore the dir_vec-direction
        component that decayed from the peak state. Return final gold rank.
        dir_vec=None means no patch (baseline). dir_vec must be a unit vector.
        """
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long, device=device)
        gold_token = sample["primary_answer_ids"][0]

        captured = {}

        def capture_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["peak"] = h[0, -1, :].detach().clone().float()
            return out

        def patch_hook(module, inp, out):
            if dir_vec is None or "peak" not in captured:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            cur = h[0, -1, :].float()
            peak = captured["peak"]
            delta = peak - cur                       # what decayed since peak
            comp = (delta @ dir_vec)                 # amount along target dir
            comp = torch.clamp(comp, min=0.0)        # only restore, never suppress
            h[0, -1, :] = (cur + beta * comp * dir_vec).to(h.dtype)
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        handles = [layers[peak_layer].register_forward_hook(capture_hook)]
        for l in range(peak_layer + 1, num_layers):
            handles.append(layers[l].register_forward_hook(patch_hook))

        with torch.no_grad():
            logits = model(input_ids, use_cache=False).logits[0, -1, :]
        for h in handles:
            h.remove()

        rank = (logits > logits[gold_token]).sum().item()
        return rank

    def eval_group(group_meta, patch_kind):
        """patch_kind in {'gold','null'}.

        gold: restore the decayed component along the gold unembedding direction.
        null: restore the decayed component along a random competitor direction
              that has been ORTHOGONALIZED against the gold direction, so the
              control cannot smuggle in any gold-direction signal. This is the
              matched control: same peak state, same restoration mechanism, same
              strength, only the direction differs and is guaranteed gold-free.
        """
        helped = 0
        recovered = 0          # patched final rank == 0
        net_recovered = 0      # baseline NOT top-1 but patch makes it top-1
        base_top1 = 0
        deltas = []
        n = 0
        for meta in tqdm(group_meta, desc=f"patch={patch_kind}"):
            sid = meta["id"]
            if sid not in prep_map:
                continue
            s = prep_map[sid]
            gold_token = s["primary_answer_ids"][0]
            gold_dir = unembed[gold_token].to(device)
            gold_dir = gold_dir / (gold_dir.norm() + 1e-6)

            base_rank = final_gold_rank_with_patch(s, meta["peak_layer"], None, 0.0)

            if patch_kind == "gold":
                dir_vec = gold_dir
            else:  # matched-null: random competitor, orthogonalized vs gold
                t = int(rng.integers(0, vocab_size))
                while t == gold_token:
                    t = int(rng.integers(0, vocab_size))
                nd = unembed[t].to(device)
                nd = nd / (nd.norm() + 1e-6)
                nd = nd - (nd @ gold_dir) * gold_dir      # remove gold component
                dir_vec = nd / (nd.norm() + 1e-6)

            patched_rank = final_gold_rank_with_patch(s, meta["peak_layer"], dir_vec, args.beta)
            d = base_rank - patched_rank  # positive = patch improved gold rank
            deltas.append(d)
            helped += int(d > 0)
            recovered += int(patched_rank == 0)
            base_top1 += int(base_rank == 0)
            net_recovered += int(base_rank != 0 and patched_rank == 0)
            n += 1
        return {"n": n, "helped": helped, "recovered": recovered,
                "net_recovered": net_recovered, "base_top1": base_top1,
                "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
                "median_delta": float(np.median(deltas)) if deltas else 0.0}

    half = args.n // 2
    pres_eval = [s for s in pres if s["id"] in prep_map][:half]
    form_eval = [s for s in form if s["id"] in prep_map][:half]

    print(f"\nGOLD patch (beta={args.beta})")
    pres_gold = eval_group(pres_eval, "gold")
    form_gold = eval_group(form_eval, "gold")
    print(f"\nNULL patch (matched control)")
    pres_null = eval_group(pres_eval, "null")

    print("\n" + "=" * 70)
    print(f"Activation Patching v4: causal peak-state restoration (beta={args.beta})")
    print("=" * 70)

    def show(tag, r):
        n = max(r["n"], 1)
        eligible = max(r["n"] - r["base_top1"], 1)  # samples not already top-1
        print(f"\n[{tag}] n={r['n']} (already top-1 at baseline: {r['base_top1']})")
        print(f"  gold-rank improved: {r['helped']}/{r['n']} ({100*r['helped']/n:.1f}%)")
        print(f"  recovered to top-1: {r['recovered']}/{r['n']} ({100*r['recovered']/n:.1f}%)")
        print(f"  NET recovered (was not top-1 -> top-1): "
              f"{r['net_recovered']}/{eligible} ({100*r['net_recovered']/eligible:.1f}%)")
        print(f"  mean rank improvement: {r['mean_delta']:.1f} (median {r['median_delta']:.0f})")

    show("preservation + GOLD patch", pres_gold)
    show("formation    + GOLD patch", form_gold)
    show("preservation + NULL patch (control)", pres_null)

    def net_rate(r):
        elig = max(r["n"] - r["base_top1"], 1)
        return r["net_recovered"] / elig

    pr = net_rate(pres_gold)
    fr = net_rate(form_gold)
    nr = net_rate(pres_null)
    print("\n[Key comparisons: NET recovery-to-top-1 (excludes already-top-1)]")
    print(f"  preservation GOLD vs formation GOLD: "
          f"{100*pr:.1f}% vs {100*fr:.1f}%"
          + (f"  ({pr/fr:.1f}x)" if fr > 0 else "  (formation=0)"))
    print(f"  preservation GOLD vs preservation NULL: "
          f"{100*pr:.1f}% vs {100*nr:.1f}%"
          + (f"  ({pr/nr:.1f}x)" if nr > 0 else "  (null=0)"))

    # ---- Significance testing on NET recovery (2x2 Fisher exact) ----
    def elig(r):
        return max(r["n"] - r["base_top1"], 0)

    fisher_gf = fisher_pn = None
    try:
        from scipy.stats import fisher_exact
        # preservation-gold vs formation-gold
        t1 = [[pres_gold["net_recovered"], elig(pres_gold) - pres_gold["net_recovered"]],
              [form_gold["net_recovered"], elig(form_gold) - form_gold["net_recovered"]]]
        _, fisher_gf = fisher_exact(t1, alternative="greater")
        # preservation-gold vs preservation-null (paired samples, same items;
        # Fisher is conservative here but reported as a sanity check)
        t2 = [[pres_gold["net_recovered"], elig(pres_gold) - pres_gold["net_recovered"]],
              [pres_null["net_recovered"], elig(pres_null) - pres_null["net_recovered"]]]
        _, fisher_pn = fisher_exact(t2, alternative="greater")
        print(f"\n  Fisher exact p (pres-gold > form-gold) = {fisher_gf:.4f}")
        print(f"  Fisher exact p (pres-gold > pres-null) = {fisher_pn:.4f}")
    except Exception as e:
        print(f"  (scipy unavailable: {e})")

    out_file = os.path.join(config.DATA_DIR, "activation_patching_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"beta": args.beta,
                   "pres_gold": pres_gold, "form_gold": form_gold,
                   "pres_null": pres_null,
                   "fisher_p_gold_vs_form": fisher_gf,
                   "fisher_p_gold_vs_null": fisher_pn}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
