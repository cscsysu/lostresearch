"""
Component-selective patching: is the LAYER-35 MLP causally responsible?

Reviewer round-6 requirement: the peak-state patch (v4) restores the gold
direction at every layer after L*, which proves the decayed gold signal is
causal but NOT that the layer-35 MLP is the component that removed it.
This experiment patches ONE component at ONE layer only.

Patch (direction-selective ablation of a single component write):
    At the MLP (or attention) output of layer l, let m be the write vector at
    the last prompt position. Remove exactly the part that points AGAINST the
    gold direction:
        m' = m - gamma * clamp_{<=0}(m . w_gold) * w_gold
    With gamma=1 the gold-suppressive part of this component's contribution
    is deleted; everything else is untouched.

Conditions (identical patch, only target differs):
    1. mlp@35      -- the attributed component
    2. mlp@33/34/36/37 -- adjacent MLPs (specificity control)
    3. attn@35     -- same-layer attention (component control)
    4. mlp@35-randorth -- random direction orthogonalized vs gold (direction control)

Metric: net recovery to top-1 (baseline NOT top-1 -> patched top-1) and mean
gold-rank change, on preservation failures.

Usage:
  python run_component_patching.py --n 150 --gamma 1.0
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
    parser.add_argument("--gammas", type=str, default="1.0,2.0,4.0",
                        help="Comma-separated ablation strengths to sweep")
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
    args = parser.parse_args()
    gammas = [float(g) for g in args.gammas.split(",")]

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
    vocab_size = unembed.shape[0]
    device = model.device

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    prep_map = {s["id"]: s for s in prepared}

    rng = np.random.default_rng(0)

    def final_gold_rank(sample, target=None):
        """target = (layer_idx, kind, gamma) or None for baseline."""
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long,
                                 device=device)
        gold_token = sample["primary_answer_ids"][0]
        w = unembed[gold_token].to(device)
        w = w / (w.norm() + 1e-6)

        if target is None:
            w_patch, gamma, module = None, 0.0, None
        else:
            l_idx, kind, gamma = target
            if kind == "mlp":
                module, w_patch = layers[l_idx].mlp, w
            elif kind == "attn":
                module, w_patch = layers[l_idx].self_attn, w
            elif kind == "mlp_randorth":
                t = int(rng.integers(0, vocab_size))
                nd = unembed[t].to(device)
                nd = nd / (nd.norm() + 1e-6)
                nd = nd - (nd @ w) * w
                module, w_patch = layers[l_idx].mlp, nd / (nd.norm() + 1e-6)
            else:
                raise ValueError(kind)

        def hook(module_, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            m = h[0, -1, :].float()
            c = torch.clamp(m @ w_patch, max=0.0)   # only the anti-gold part
            h[0, -1, :] = (m - gamma * c * w_patch).to(h.dtype)
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        handle = module.register_forward_hook(hook) if module is not None else None
        with torch.no_grad():
            logits = model(input_ids, use_cache=False).logits[0, -1, :].float()
        if handle is not None:
            handle.remove()
        return int((logits > logits[gold_token]).sum().item())

    eval_set = [s for s in pres if s["id"] in prep_map][:args.n]
    print(f"Evaluating {len(eval_set)} preservation failures")
    print(f"Model has {num_layers} layers (indices 0-{num_layers-1}); "
          f"layer 35 is the FINAL layer")

    # Layer 35 is the last layer, so "adjacent" controls must be BELOW it.
    mlp_neighbors = [l for l in (31, 32, 33, 34) if l < num_layers]
    conditions = [("baseline", None)]
    for g in gammas:
        conditions += [(f"mlp@35_g{g:g}", (35, "mlp", g))]
        conditions += [(f"mlp@{l}_g{g:g}", (l, "mlp", g)) for l in mlp_neighbors]
        conditions += [(f"attn@35_g{g:g}", (35, "attn", g))]
        conditions += [(f"mlp@35-randorth_g{g:g}", (35, "mlp_randorth", g))]

    results = {}
    base_ranks = None
    for name, target in conditions:
        ranks = []
        for meta in tqdm(eval_set, desc=name):
            s = prep_map[meta["id"]]
            ranks.append(final_gold_rank(s, target))
        ranks = np.array(ranks)
        if name == "baseline":
            base_ranks = ranks
            results[name] = {"n": len(ranks),
                             "base_top1": int((ranks == 0).sum())}
            continue
        l_idx, kind, gamma = target
        elig = base_ranks != 0
        net_rec = int(((ranks == 0) & elig).sum())
        results[name] = {
            "n": len(ranks), "gamma": gamma,
            "net_recovered": net_rec,
            "eligible": int(elig.sum()),
            "net_rate": net_rec / max(int(elig.sum()), 1),
            "mean_rank_improve": float((base_ranks - ranks).mean()),
        }
        print(f"  {name:22s} net={net_rec}/{int(elig.sum())} "
              f"({100*net_rec/max(int(elig.sum()),1):.1f}%)  "
              f"mean dRank={results[name]['mean_rank_improve']:+.1f}")

    # Significance: mlp@35 vs controls at each gamma (Fisher on net recovery)
    try:
        from scipy.stats import fisher_exact
        for g in gammas:
            ref = results[f"mlp@35_g{g:g}"]
            for name in [f"mlp@{l}_g{g:g}" for l in mlp_neighbors] + \
                        [f"attn@35_g{g:g}", f"mlp@35-randorth_g{g:g}"]:
                c = results[name]
                table = [[ref["net_recovered"], ref["eligible"] - ref["net_recovered"]],
                         [c["net_recovered"], c["eligible"] - c["net_recovered"]]]
                _, p = fisher_exact(table, alternative="greater")
                results[name]["fisher_p_vs_mlp35"] = float(p)
                print(f"  Fisher mlp@35 > {name}: p = {p:.4f}")
    except Exception as e:
        print(f"  (scipy unavailable: {e})")

    out_file = os.path.join(config.DATA_DIR, "component_patching_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"gammas": gammas, "results": results}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
