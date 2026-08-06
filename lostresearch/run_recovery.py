"""
Recovery experiment v2 (adaptive / logit-level restore).

v1 added a fixed alpha * (peak projection) * w_gold to the residual stream;
recovery rose with alpha (3.3% -> 16.7%) but stayed modest, and the fixed boost
does not track per-layer decay.

v2 adapts per layer: if the gold projection LENGTH along the unit gold direction
has dropped below its peak, inject exactly the gap to hold it at the peak (or
`scale` times the peak). All quantities are projection lengths along a unit
vector, so the injected vector has norm == the gap -- correct units.

We scan `scale` so one run covers "restore to peak" (1.0) and over-restoration
(2.0, 4.0), reusing hiddens and generations.

Controls (at the largest scale):
  - random direction (same gap logic) -> should not help
  - gold restore only in shallow layers (l < l_peak) -> layer-specificity

Usage:
  python run_recovery.py --n 60 --scales 1.0,2.0,4.0
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
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--scales", type=str, default="1.0,2.0,4.0")
    args = parser.parse_args()
    scales = [float(x) for x in args.scales.split(",")]

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
    print(f"Errors available: {len(errors)}; will evaluate up to {args.n}")

    # ---- per-scale counters ----
    gold_rec = {sc: 0 for sc in scales}
    gold_tot = {sc: 0 for sc in scales}
    rand_rec = 0
    n_eval = 0
    results = []

    for s in tqdm(errors[:args.n], desc="Recovery (adaptive)"):
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        aliases = s["aliases"]
        gold_w = unembed_f[gold_token].to(device)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        base_out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if is_answer_correct(base_text, aliases):
            continue

        # collect per-layer hiddens -> gold projection per layer
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

        # Peak gold projection LENGTH along the unit gold direction (not the raw
        # dot product with the large unembedding vector).
        gold_unit = gold_w / (gold_w.norm().item() + 1e-8)
        proj = []
        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()
            proj.append(torch.dot(h, gold_unit).item())  # projection length
        mid = num_layers // 2
        l_peak = int(np.argmax(proj[:mid])) if mid > 0 else 0
        peak_val = proj[l_peak]

        # Adaptive restore: at each affected layer, if the projection LENGTH of h
        # along the unit `direction` has dropped below `target`, inject the gap so
        # it is held at that level -- a per-layer repair with no strength knob.
        # All quantities are projection lengths along a unit vector, so the
        # injected vector has norm == the gap (correct units).
        def make_restore(layer_pred, direction_unit, target, scale=1.0):
            d = direction_unit
            def hook_fn(layer_idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    rest = output[1:] if isinstance(output, tuple) else ()
                    if layer_pred(layer_idx):
                        hf = h[0, -1, :].float()
                        cur = torch.dot(hf, d).item()
                        if cur < target * scale:
                            gap = (target * scale - cur)
                            h[0, -1, :] = h[0, -1, :] + gap * d.to(h.device).to(h.dtype)
                    if rest:
                        return (h,) + rest
                    return h
                return hook
            return hook_fn

        def generate(hook_factory):
            hooks = [layers[l].register_forward_hook(hook_factory(l)) for l in range(num_layers)]
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            for hk in hooks:
                hk.remove()
            return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        shallow_end = max(2, l_peak // 2)
        max_scale = max(scales)

        # 1) adaptive gold restore in the decay region [l_peak, L), per scale
        per_gold = {}
        for sc in scales:
            gen = generate(make_restore(lambda li: li >= l_peak, gold_unit, peak_val, scale=sc))
            rec = is_answer_correct(gen, aliases)
            per_gold[sc] = rec
            gold_tot[sc] += 1
            gold_rec[sc] += int(rec)

        # 2) adaptive gold restore only in shallow layers [0, shallow_end) at max scale
        gs = generate(make_restore(lambda li: li < shallow_end, gold_unit, peak_val, scale=max_scale))
        s_rec = is_answer_correct(gs, aliases)

        # 3) random-direction adaptive restore (same gap logic on a unit random dir)
        rand_dir = torch.randn_like(gold_w)
        rand_unit = rand_dir / (rand_dir.norm().item() + 1e-8)
        gr = generate(make_restore(lambda li: li >= l_peak, rand_unit, peak_val, scale=max_scale))
        r_rec = is_answer_correct(gr, aliases)
        rand_rec += int(r_rec)

        n_eval += 1
        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "l_peak": l_peak, "peak_val": peak_val,
            "gold": {str(sc): per_gold[sc] for sc in scales},
            "random_recover": r_rec,
            "shallow_recover": s_rec,
        })

    print("\n" + "=" * 70)
    print(f"Adaptive (logit-level) recovery, errors evaluated: {n_eval}")
    print("=" * 70)
    for sc in scales:
        if gold_tot[sc] > 0:
            print(f"  Gold restore at decay region (scale={sc}): "
                  f"{gold_rec[sc]}/{gold_tot[sc]} ({100*gold_rec[sc]/max(gold_tot[sc],1):.1f}%)")
    print(f"  Random restore control (scale={max_scale}):       "
          f"{rand_rec}/{n_eval} ({100*rand_rec/max(n_eval,1):.1f}%)")
    print(f"  Gold restore shallow layers (scale={max_scale}):  "
          f"{s_rec}/{n_eval} ({100*s_rec/max(n_eval,1):.1f}%)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"mode": "adaptive", "scales": scales, "results": results}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
