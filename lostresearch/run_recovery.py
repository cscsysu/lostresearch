"""
Recovery experiment v2 (logit-level / adaptive): hold the gold signal at its
peak across the decay region.

v1 added a fixed alpha * (peak projection) * w_gold to the residual stream.
Recovery rose with alpha (3.3% -> 16.7%) but stayed modest. The fixed boost is
blunt: it does not track the per-layer decay.

v2 adapts: for each layer l >= l_peak, we compute the current gold projection
onto the gold unembedding direction and, if it has dropped below its peak,
inject exactly enough of w_gold to restore it to the peak value. This keeps the
gold-answer signal constant at its strongest level through the region where it
would otherwise decay -- a direct, per-layer 'logit-level' repair of the
decay, with no free strength parameter.

Controls (as before):
  - random direction (restore a random direction to its peak) -> should not help
  - gold restoration only in shallow layers (l < l_peak)         -> layer-specificity

Usage:
  python run_recovery.py --n 60
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
    args = parser.parse_args()

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

    # ---- baseline pass: collect per-layer gold projection + peak + generated text ----
    gold_rec = shallow_rec = rand_rec = 0
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

        proj = []
        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()
            proj.append(torch.dot(h, gold_w).item())
        mid = num_layers // 2
        l_peak = int(np.argmax(proj[:mid])) if mid > 0 else 0
        peak_val = proj[l_peak]

        # Adaptive restore: at each affected layer, if the projection of h along
        # `direction` has dropped below `target`, inject the gap so it is kept at
        # that level -- a per-layer repair with no free strength parameter.
        def make_restore(layer_pred, direction, target, scale=1.0):
            d = direction / (direction.norm().item() + 1e-8)
            def hook_fn(layer_idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    rest = output[1:] if isinstance(output, tuple) else ()
                    if layer_pred(layer_idx):
                        hf = h[0, -1, :].float()
                        cur = torch.dot(hf, direction).item()
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

        # 1) adaptive gold restore in the decay region [l_peak, L)
        g = generate(make_restore(lambda li: li >= l_peak, gold_w, peak_val))
        g_rec = is_answer_correct(g, aliases)
        gold_rec += int(g_rec)

        # 2) adaptive gold restore only in shallow layers [0, shallow_end)
        gs = generate(make_restore(lambda li: li < shallow_end, gold_w, peak_val))
        s_rec = is_answer_correct(gs, aliases)
        shallow_rec += int(s_rec)

        # 3) random-direction adaptive restore (same gap logic on a random dir)
        rand_dir = torch.randn_like(gold_w)
        gr = generate(make_restore(lambda li: li >= l_peak, rand_dir, peak_val))
        r_rec = is_answer_correct(gr, aliases)
        rand_rec += int(r_rec)

        n_eval += 1
        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "l_peak": l_peak, "peak_val": peak_val,
            "gold_recover": g_rec,
            "random_recover": r_rec,
            "shallow_recover": s_rec,
        })

    print("\n" + "=" * 70)
    print(f"Adaptive (logit-level) recovery, errors evaluated: {n_eval}")
    print("=" * 70)
    print(f"  Gold restore at decay region: {gold_rec}/{n_eval} ({100*gold_rec/max(n_eval,1):.1f}%)")
    print(f"  Random restore control:       {rand_rec}/{n_eval} ({100*rand_rec/max(n_eval,1):.1f}%)")
    print(f"  Gold restore shallow layers:  {shallow_rec}/{n_eval} ({100*shallow_rec/max(n_eval,1):.1f}%)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"mode": "adaptive", "results": results}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
