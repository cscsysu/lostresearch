"""
Prediction-guided intervention: does the trajectory tell us WHICH errors are
repairable, and WHERE to repair them?

Reviewer priority: connect prediction to intervention. Our trajectory analysis
splits errors into
  - preservation failures: the gold answer WAS competitive (rank <= k) at some
    intermediate layer but lost its advantage by the output;
  - formation failures:    the gold answer was never competitive.
The mechanistic prediction is that late-layer steering should repair the first
group and not the second, because in a preservation failure the signal exists
and only needs to be retained.

Intervention (activation steering, unconditional so it always acts):
  at layers l >= l_decay:  h <- h + beta * ||h|| * unit(w_gold)
The boost is a fixed fraction beta of the local residual norm, so it is
comparable across depths (the residual norm grows with depth, which is exactly
what broke an earlier absolute-projection version of this experiment).
l_decay is the layer of best (lowest) intermediate gold rank -- the point after
which the signal decays.

Controls (at the largest beta):
  - random unit direction, same magnitude  -> tests direction specificity
  - steering only in early layers [0, l_decay/2) -> tests layer specificity

Headline comparison: recovery rate for preservation vs formation failures.

Usage:
  python run_recovery.py --n 80 --betas 0.1,0.3,0.5
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
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--betas", type=str, default="0.1,0.3,0.5")
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
    args = parser.parse_args()
    betas = [float(x) for x in args.betas.split(",")]
    max_beta = max(betas)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    final_norm = model.model.norm
    unembed = model.lm_head.weight
    unembed_f = unembed.float()
    device = model.device

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
    print(f"Errors available: {len(errors)}; evaluating up to {args.n}")

    # counters: group -> beta -> recovered / total
    groups = ["preservation", "formation"]
    rec = {g: {b: 0 for b in betas} for g in groups}
    tot = {g: 0 for g in groups}
    ctrl_rand = {g: 0 for g in groups}
    ctrl_shallow = {g: 0 for g in groups}
    diag = []
    results = []

    for s in tqdm(errors[:args.n], desc="Prediction-guided steering"):
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        aliases = s["aliases"]
        gold_w = unembed_f[gold_token].to(device)
        gold_unit = gold_w / (gold_w.norm().item() + 1e-8)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        base_out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:],
                                     skip_special_tokens=True).strip()
        if is_answer_correct(base_text, aliases):
            continue

        #---- collect per-layer hiddens, gold rank, cosine alignment ----
        hidden_buffer = {}

        def make_hook(idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                hidden_buffer[idx] = h[0, -1, :].detach().clone()
            return hook

        hooks = [layer.register_forward_hook(make_hook(l)) for l, layer in enumerate(layers)]
        with torch.no_grad():
            model(input_ids, use_cache=False)
        for h in hooks:
            h.remove()

        ranks, cosines, projs, norms = [], [], [], []
        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            ranks.append((lr > lr[gold_token]).sum().item())
            projs.append(torch.dot(h, gold_unit).item())
            norms.append(h.norm().item())
            cosines.append(projs[-1] / (norms[-1] + 1e-8))

        inter_ranks = ranks[:-1] if num_layers > 1 else ranks
        best_rank = min(inter_ranks)
        l_decay = int(np.argmin(inter_ranks))
        group = "preservation" if best_rank <= args.k else "formation"

        # ---- steering intervention (unconditional; fraction of local norm) ----
        def make_steer(layer_pred, direction_unit, beta):
            def hook_fn(layer_idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    rest = output[1:] if isinstance(output, tuple) else ()
                    if layer_pred(layer_idx):
                        cur = h[0, -1, :]
                        scale = beta * cur.norm()
                        h[0, -1, :] = cur + scale * direction_unit.to(h.device).to(h.dtype)
                    if rest:
                        return (h,) + rest
                    return h
                return hook
            return hook_fn

        def generate(hook_factory):
            hks = [layers[l].register_forward_hook(hook_factory(l)) for l in range(num_layers)]
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            for hk in hks:
                hk.remove()
            return tokenizer.decode(out[0][input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()

        tot[group] += 1
        per_beta = {}
        for b in betas:
            gen = generate(make_steer(lambda li: li >= l_decay, gold_unit, b))
            ok = is_answer_correct(gen, aliases)
            per_beta[b] = ok
            rec[group][b] += int(ok)

        # control 1: random direction at max beta
        rand_unit = torch.randn_like(gold_unit)
        rand_unit = rand_unit / (rand_unit.norm() + 1e-8)
        gr = generate(make_steer(lambda li: li >= l_decay, rand_unit, max_beta))
        r_ok = is_answer_correct(gr, aliases)
        ctrl_rand[group] += int(r_ok)

        # control 2: gold steering only in early layers
        shallow_end = max(1, l_decay // 2)
        gs = generate(make_steer(lambda li: li < shallow_end, gold_unit, max_beta))
        s_ok = is_answer_correct(gs, aliases)
        ctrl_shallow[group] += int(s_ok)

        diag.append({"best_rank": best_rank, "l_decay": l_decay,
                     "cos_peak": cosines[l_decay], "cos_final": cosines[-1],
                     "norm_peak": norms[l_decay], "norm_final": norms[-1]})
        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "group": group, "best_rank": best_rank, "l_decay": l_decay,
            "steer": {str(b): per_beta[b] for b in betas},
            "random_control": r_ok, "shallow_control": s_ok,
        })

    # ---- report ----
    print("\n" + "=" * 74)
    print("Prediction-guided steering: preservation vs formation failures")
    print("=" * 74)
    for g in groups:
        if tot[g] == 0:
            print(f"\n[{g}] no samples")
            continue
        print(f"\n[{g}]  n={tot[g]}")
        for b in betas:
            print(f"  steer at decay layer (beta={b}): "
                  f"{rec[g][b]}/{tot[g]} ({100*rec[g][b]/tot[g]:.1f}%)")
        print(f"  random-direction control (beta={max_beta}): "
              f"{ctrl_rand[g]}/{tot[g]} ({100*ctrl_rand[g]/tot[g]:.1f}%)")
        print(f"  early-layer control (beta={max_beta}):      "
              f"{ctrl_shallow[g]}/{tot[g]} ({100*ctrl_shallow[g]/tot[g]:.1f}%)")

    if diag:
        print("\n--- diagnostics (medians) ---")
        print(f"  best intermediate gold rank: {np.median([d['best_rank'] for d in diag]):.0f}")
        print(f"  decay layer l_decay:         {np.median([d['l_decay'] for d in diag]):.0f}")
        print(f"  cos(h, w_gold) at l_decay:   {np.median([d['cos_peak'] for d in diag]):.4f}")
        print(f"  cos(h, w_gold) at final:     {np.median([d['cos_final'] for d in diag]):.4f}")
        print(f"  ||h|| at l_decay:            {np.median([d['norm_peak'] for d in diag]):.1f}")
        print(f"  ||h|| at final:              {np.median([d['norm_final'] for d in diag]):.1f}")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"betas": betas, "k": args.k, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
