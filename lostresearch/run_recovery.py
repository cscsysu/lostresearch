"""
Recovery experiment (causal sufficiency): does restoring the gold signal at the
decaying layer turn a wrong answer correct?

Reviewer concern: the ablation shows the gold direction is NECESSARY (removing it
flips correct -> wrong) but we never show SUFFICIENCY -- restoring the decayed
trajectory makes an error correct.

We sweep the boost strength alpha to distinguish a strength problem from a true
null result. For each error sample:
  - find the gold peak layer l_peak in the first half of the network
  - inject a boost + alpha * (gold projection at l_peak) along w_gold at
    layers >= l_peak, regenerate
Controls (at the largest alpha):
  - random direction (same norm) at l_peak..end  -> should NOT recover
  - gold boost only in shallow layers (< l_peak)  -> layer-specificity test

Usage:
  python run_recovery.py --n 60 --alphas 0.5,1.0,2.0,4.0
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
    parser.add_argument("--alphas", type=str, default="0.5,1.0,2.0,4.0")
    args = parser.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

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

    def generate(hook_fn, prompt_ids):
        hooks = []
        for l in range(num_layers):
            hooks.append(layers[l].register_forward_hook(hook_fn(l)))
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for h in hooks:
            h.remove()
        return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Per-alpha counters.
    gold_rec = {a: 0 for a in alphas}
    gold_tot = {a: 0 for a in alphas}
    n_rand = n_shallow = 0
    n_rand_rec = n_shallow_rec = 0

    results = []
    max_alpha = max(alphas)

    for s in tqdm(errors[:args.n], desc="Recovery"):
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

        support = []
        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            support.append(lr[gold_token].item())
        mid = num_layers // 2
        l_peak = int(np.argmax(support[:mid])) if mid > 0 else 0
        peak_proj_norm = torch.dot(hidden_buffer[l_peak].to(device).float(), gold_w).abs().item()
        unit = gold_w / (gold_w.norm().item() + 1e-8)
        # vector that, added along w_gold, re-adds `alpha` x the peak projection.
        inj = {a: (a * peak_proj_norm) * unit for a in alphas}

        def make_inject(boost_dir, start_layer, end_layer):
            def hook_fn(layer_idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    rest = output[1:] if isinstance(output, tuple) else ()
                    if start_layer <= layer_idx < end_layer:
                        h[0, -1, :] = h[0, -1, :] + boost_dir.to(h.device).to(h.dtype)
                    if rest:
                        return (h,) + rest
                    return h
                return hook
            return hook_fn

        # gold boost across alphas (at decay layer -> end)
        per_gold = {}
        for a in alphas:
            gen = generate(make_inject(inj[a], l_peak, num_layers), prompt_ids)
            rec = is_answer_correct(gen, aliases)
            per_gold[a] = rec
            gold_tot[a] += 1
            gold_rec[a] += int(rec)

        # random control at max alpha
        rand_dir = torch.randn_like(inj[max_alpha])
        rand_dir = rand_dir / (rand_dir.norm() + 1e-8) * inj[max_alpha].norm()
        rand_gen = generate(make_inject(rand_dir, l_peak, num_layers), prompt_ids)
        rand_rec = is_answer_correct(rand_gen, aliases)
        n_rand += 1
        n_rand_rec += int(rand_rec)

        # shallow-layer gold boost at max alpha (layer-specificity)
        shallow_end = max(2, l_peak // 2)
        sh_gen = generate(make_inject(inj[max_alpha], 0, shallow_end), prompt_ids)
        sh_rec = is_answer_correct(sh_gen, aliases)
        n_shallow += 1
        n_shallow_rec += int(sh_rec)

        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "l_peak": l_peak,
            "gold": {str(a): per_gold[a] for a in alphas},
            "random_recover": rand_rec,
            "shallow_recover": sh_rec,
        })

    print("\n" + "=" * 70)
    print(f"Recovery sweep (alphas={alphas}), errors evaluated: {len(results)}")
    print("=" * 70)
    for a in alphas:
        if gold_tot[a] > 0:
            print(f"  Gold boost at decay layer (alpha={a}): "
                  f"{gold_rec[a]}/{gold_tot[a]} ({100*gold_rec[a]/gold_tot[a]:.1f}%)")
    print(f"  Random boost control (alpha={max_alpha}): "
          f"{n_rand_rec}/{n_rand} ({100*n_rand_rec/max(n_rand,1):.1f}%)")
    print(f"  Gold boost shallow layers (alpha={max_alpha}): "
          f"{n_shallow_rec}/{n_shallow} ({100*n_shallow_rec/max(n_shallow,1):.1f}%)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"alphas": alphas, "results": results}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
