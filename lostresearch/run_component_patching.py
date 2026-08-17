"""
Component-selective patching v2: is the LAYER-35 MLP causally responsible?

v1 design flaw (why mlp@35 showed zero effect):
  v1 removed only the write component along the GOLD unembedding direction,
  clamp(m.w_gold, max=0). But DLA attributes layer-35 MLP a negative
  gold-vs-COMPETITOR margin contribution: the MLP can suppress the margin by
  boosting the competitor while writing zero-to-positive along gold. In that
  case the clamp removes nothing and the patch is a no-op. mlp@33's positive
  v1 effect likely reflects incidental negative gold projections.

v2 fixes:
  PRIMARY direction = contrast d = normalize(w_gold - w_competitor), the same
  quantity DLA attributes. We remove the margin-suppressive component of the
  write: m' = m - gamma * clamp(m.d, max=0) * d.
  Controls: adjacent MLPs (31-34), same-layer attention, random direction
  orthogonalized vs d, plus a classic full ZERO-ablation of single MLP
  writes (the standard, operationalization-free causal test).

Usage:
  python run_component_patching.py --n 150
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
    parser.add_argument("--gammas", type=str, default="1.0,2.0,4.0")
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
    existing = {r["id"]: r for r in all_results}

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

    def competitor_token(meta, prep):
        """First token of the model's generated (wrong) answer."""
        gen = existing.get(meta["id"], {}).get("generated", "")
        if gen:
            ids = tokenizer.encode(gen, add_special_tokens=False)
            if ids:
                return ids[0]
        return None

    def final_gold_rank(sample, comp_tok, target=None):
        """target = (layer_idx, kind, gamma) or None for baseline.

        kinds: 'contrast'  remove margin-suppressive write along
                            d = w_gold - w_competitor
               'randorth'  same removal along a random dir orthogonal to d
               'zero'      full zero-ablation of the component write
        """
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long,
                                 device=device)
        gold_token = sample["primary_answer_ids"][0]
        wg = unembed[gold_token].to(device)
        wg = wg / (wg.norm() + 1e-6)
        d = wg
        if comp_tok is not None:
            wc = unembed[comp_tok].to(device)
            d = wg - wc / (wc.norm() + 1e-6)
            d = d / (d.norm() + 1e-6)

        if target is None:
            module, w_patch, gamma, zero = None, None, 0.0, False
        else:
            l_idx, kind, gamma = target
            zero = kind == "zero"
            if kind in ("contrast", "zero"):
                module, w_patch = layers[l_idx].mlp, d
            elif kind == "attn_contrast":
                module, w_patch = layers[l_idx].self_attn, d
            elif kind == "randorth":
                t = int(rng.integers(0, vocab_size))
                nd = unembed[t].to(device)
                nd = nd / (nd.norm() + 1e-6)
                nd = nd - (nd @ d) * d
                module, w_patch = layers[l_idx].mlp, nd / (nd.norm() + 1e-6)
            else:
                raise ValueError(kind)

        def hook(module_, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            m = h[0, -1, :].float()
            if zero:
                h[0, -1, :] = 0.0
            else:
                c = torch.clamp(m @ w_patch, max=0.0)  # margin-suppressive part
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
    comp_tokens = {}
    for meta in eval_set:
        comp_tokens[meta["id"]] = competitor_token(meta, prep_map[meta["id"]])
    print(f"Evaluating {len(eval_set)} preservation failures; "
          f"competitor token available for "
          f"{sum(v is not None for v in comp_tokens.values())}")

    mlp_neighbors = [l for l in (31, 32, 33, 34) if l < num_layers]
    conditions = [("baseline", None)]
    for g in gammas:
        conditions += [(f"mlp@35_contrast_g{g:g}", (35, "contrast", g))]
        conditions += [(f"mlp@{l}_contrast_g{g:g}", (l, "contrast", g))
                       for l in mlp_neighbors]
        conditions += [(f"attn@35_contrast_g{g:g}", (35, "attn_contrast", g))]
        conditions += [(f"mlp@35_randorth_g{g:g}", (35, "randorth", g))]
    # classic zero-ablation (gamma-independent; run once per layer)
    conditions += [(f"mlp@{l}_zero", (l, "zero", 1.0))
                   for l in [33, 34, 35] if l < num_layers]

    results = {}
    base_ranks = None
    for name, target in conditions:
        ranks = []
        for meta in tqdm(eval_set, desc=name):
            s = prep_map[meta["id"]]
            ranks.append(final_gold_rank(s, comp_tokens[meta["id"]], target))
        ranks = np.array(ranks)
        if name == "baseline":
            base_ranks = ranks
            results[name] = {"n": len(ranks),
                             "base_top1": int((ranks == 0).sum())}
            continue
        elig = base_ranks != 0
        net_rec = int(((ranks == 0) & elig).sum())
        results[name] = {
            "n": len(ranks), "gamma": target[2] if target else None,
            "net_recovered": net_rec,
            "eligible": int(elig.sum()),
            "net_rate": net_rec / max(int(elig.sum()), 1),
            "mean_rank_improve": float((base_ranks - ranks).mean()),
        }
        print(f"  {name:26s} net={net_rec}/{int(elig.sum())} "
              f"({100*net_rec/max(int(elig.sum()),1):.1f}%)  "
              f"mean dRank={results[name]['mean_rank_improve']:+.1f}")

    try:
        from scipy.stats import fisher_exact
        for g in gammas:
            ref = results[f"mlp@35_contrast_g{g:g}"]
            ctrls = [f"mlp@{l}_contrast_g{g:g}" for l in mlp_neighbors] + \
                    [f"attn@35_contrast_g{g:g}", f"mlp@35_randorth_g{g:g}"]
            for name in ctrls:
                c = results[name]
                table = [[ref["net_recovered"], ref["eligible"] - ref["net_recovered"]],
                         [c["net_recovered"], c["eligible"] - c["net_recovered"]]]
                _, p = fisher_exact(table, alternative="greater")
                results[name]["fisher_p_vs_mlp35"] = float(p)
                print(f"  Fisher mlp@35 > {name}: p = {p:.4f}")
        # zero-ablation comparisons
        if "mlp@35_zero" in results:
            ref = results["mlp@35_zero"]
            for name in ["mlp@33_zero", "mlp@34_zero"]:
                if name in results:
                    c = results[name]
                    table = [[ref["net_recovered"], ref["eligible"] - ref["net_recovered"]],
                             [c["net_recovered"], c["eligible"] - c["net_recovered"]]]
                    _, p = fisher_exact(table, alternative="greater")
                    results[name]["fisher_p_vs_mlp35zero"] = float(p)
                    print(f"  Fisher mlp@35_zero > {name}: p = {p:.4f}")
    except Exception as e:
        print(f"  (scipy unavailable: {e})")

    out_file = os.path.join(config.DATA_DIR, "component_patching_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"gammas": gammas, "results": results}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
