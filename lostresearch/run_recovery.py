"""
Recovery experiment (causal sufficiency): does restoring the gold signal at the
decaying layer turn a wrong answer correct?

Reviewer concern: the ablation shows the gold direction is NECESSARY (removing it
flips correct -> wrong). But we never show SUFFICIENCY -- i.e. that repairing the
decayed trajectory makes a wrong answer correct. This is the missing causal link
between trajectory dynamics and behavior.

Design (clean, no donor samples):
  For each error sample:
    1. Run baseline, keep only samples that are actually wrong.
    2. Collect per-layer hidden states; find the "gold peak layer" l_peak where
       the gold-answer support is maximal in the first half of the network.
    3. At layers l >= l_peak, inject a fixed boost along the gold unembedding
       direction:  h' = h + alpha * w_gold.
    4. Regenerate; measure how many turn correct.
  Controls:
    - random direction (same norm) instead of w_gold  -> should NOT recover
    - boost only in the SHALLOW layers (l < l_peak)   -> tests layer-specificity
  alpha is set so the added vector has norm proportional to the gold projection
  norm at the peak layer (a matched, modest boost, not an answer injection).

Usage:
  python run_recovery.py [--n 60] [--alpha 0.5]
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
    parser.add_argument("--alpha", type=float, default=0.5)
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

    # Which samples are errors? Attach final_correct from existing results.
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

    # Storage
    results = []
    n_gold = n_rand = n_shallow = 0
    n_gold_recover = n_rand_recover = n_shallow_recover = 0

    for s in tqdm(errors[:args.n], desc="Recovery"):
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        aliases = s["aliases"]
        gold_w = unembed_f[gold_token].to(device)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # baseline (no intervention)
        base_out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if is_answer_correct(base_text, aliases):
            continue  # actually correct, skip

        # collect per-layer hiddens to find gold peak layer
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

        # gold support per layer (raw logit lens); find peak in first half
        support = []
        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            support.append(lr[gold_token].item())
        mid = num_layers // 2
        l_peak = int(np.argmax(support[:mid])) if mid > 0 else 0
        # scale alpha so the injected vector has norm ~ gold projection at peak
        peak_proj_norm = torch.dot(hidden_buffer[l_peak].to(device).float(), gold_w).abs().item()
        boost = (args.alpha * peak_proj_norm) / (gold_w.norm().item() + 1e-8)
        inj_vec = boost * gold_w

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

        # 1) gold boost at l_peak..end
        gold_gen = generate(make_inject(inj_vec, l_peak, num_layers), prompt_ids)
        gold_rec = is_answer_correct(gold_gen, aliases)
        n_gold += 1
        n_gold_recover += int(gold_rec)

        # 2) random direction control (same norm as inj_vec)
        rand_dir = torch.randn_like(inj_vec)
        rand_dir = rand_dir / (rand_dir.norm() + 1e-8) * inj_vec.norm()
        rand_gen = generate(make_inject(rand_dir, l_peak, num_layers), prompt_ids)
        rand_rec = is_answer_correct(rand_gen, aliases)
        n_rand += 1
        n_rand_recover += int(rand_rec)

        # 3) gold boost only in SHALLOW layers (before peak) -- layer-specificity
        shallow_end = max(2, l_peak // 2)
        sh_gen = generate(make_inject(inj_vec, 0, shallow_end), prompt_ids)
        sh_rec = is_answer_correct(sh_gen, aliases)
        n_shallow += 1
        n_shallow_recover += int(sh_rec)

        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "l_peak": l_peak, "boost": boost,
            "gold_recover": gold_rec, "random_recover": rand_rec, "shallow_recover": sh_rec,
        })

    print("\n" + "=" * 70)
    print(f"Recovery experiment (alpha={args.alpha}), errors evaluated: {len(results)}")
    print("=" * 70)
    print(f"  Gold boost at decay layer:   {n_gold_recover}/{n_gold} recovered ({100*n_gold_recover/n_gold:.1f}%)")
    print(f"  Random boost control:        {n_rand_recover}/{n_rand} recovered ({100*n_rand_recover/n_rand:.1f}%)")
    print(f"  Gold boost (shallow layers): {n_shallow_recover}/{n_shallow} recovered ({100*n_shallow_recover/n_shallow:.1f}%)")
    if n_gold_recover > n_rand_recover and n_gold_recover > n_shallow_recover:
        print("  => Restoring gold at the decay layer is specifically effective")
    else:
        print("  => Recovery not layer-specific (check alpha / layer selection)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"alpha": args.alpha, "results": results}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
