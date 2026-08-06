"""
Decay repair guided by the trajectory (Prediction -> Intervention).

Prior steering (adding gold-unembedding direction to the residual stream)
recovered almost nothing, because error samples' gold-answer cosine alignment
is very low and residual steering does not directly raise the gold token's rank.

This version does a *measured* repair of the decay: it restores the gold token's
final-layer logit to its OWN historical intermediate peak value. This is not
"revealing the answer" (we do not force gold to the top); it is undoing the
decay by holding the signal at the level it once reached.

Prediction step: split errors by the trajectory into
  - preservation failures: gold was competitive (rank <= k) at some intermediate
    layer -> the signal existed; its peak logit was high, so restoring to the
    peak is a targeted repair of the decay.
  - formation failures: gold was never competitive -> its peak logit was never
    high; restoring to the peak leaves it below the top, so repair does not help.

Headline test: under the same "restore gold logit to its peak" repair,
preservation failures should recover far more than formation failures. That is
the trajectory predicting repairability -- the Prediction -> Intervention link.

Implementation: at the final layer's residual output, add exactly the vector
needed to raise the gold logit to its intermediate peak logit. Because the
addition happens before final norm, the achieved logit change is monotonic in
the target; we use the intermediate peak gold logit measured with the same
logit lens.

Controls:
  - random-token "repair": restore an irrelevant token's logit to the same
    peak magnitude instead of gold.
  - a "shallow-peak" repair using only the first few layers' gold peak (which
    is low even for preservation failures) is not needed; the random control
    already tests direction specificity.

Usage:
  python run_recovery.py --n 100 --k 5
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
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
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
    unembed_f = model.lm_head.weight.float()
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

    groups = ["preservation", "formation"]
    rec = {g: 0 for g in groups}
    tot = {g: 0 for g in groups}
    rand_ctrl = {g: 0 for g in groups}
    results = []

    for s in tqdm(errors[:args.n], desc="Decay repair"):
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        aliases = s["aliases"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        base_out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:],
                                     skip_special_tokens=True).strip()
        if is_answer_correct(base_text, aliases):
            continue

        # collect per-layer hiddens -> gold logit (logit lens) + gold rank
        hidden_buffer = {}

        def make_hook(idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                hidden_buffer[idx] = h[0, -1, :].detach().clone()
            return hook

        hks = [layer.register_forward_hook(make_hook(l)) for l, layer in enumerate(layers)]
        with torch.no_grad():
            model(input_ids, use_cache=False)
        for h in hks:
            h.remove()

        gold_logits, ranks = [], []
        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            gold_logits.append(lr[gold_token].item())
            ranks.append((lr > lr[gold_token]).sum().item())
        inter_ranks = ranks[:-1] if len(ranks) > 1 else ranks
        best_rank = min(inter_ranks)
        # decay layer = argmax gold logit over intermediate layers
        l_peak = int(np.argmax(gold_logits[:-1])) if len(gold_logits) > 1 else 0
        peak_gold_logit = gold_logits[l_peak]
        final_gold_logit = gold_logits[-1]
        # how much to add to restore final logit to the peak
        gap = peak_gold_logit - final_gold_logit
        gold_w = unembed_f[gold_token].to(device)

        group = "preservation" if best_rank <= args.k else "formation"
        tot[group] += 1

        # repair: add (gap / ||w_gold||^2) * w_gold at the final layer's output,
        # which raises the gold logit by (approximately) `gap` through final head.
        add_vec = (max(gap, 0.0) / (gold_w.norm().item() ** 2 + 1e-8)) * gold_w
        last = num_layers - 1

        def make_repair_hook(vec):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                rest = output[1:] if isinstance(output, tuple) else ()
                h[0, -1, :] = h[0, -1, :].float() + vec.to(h.device).to(h.dtype)
                if rest:
                    return (h,) + rest
                return h
            return hook

        def generate_with(hook):
            hk = layers[last].register_forward_hook(hook)
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            hk.remove()
            return tokenizer.decode(out[0][input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()

        # gold repair: restore final gold logit to its intermediate peak
        if gap > 0:
            gen = generate_with(make_repair_hook(add_vec))
            ok = is_answer_correct(gen, aliases)
        else:
            # no decay (final >= peak); repair is a no-op
            ok = False
        rec[group] += int(ok)

        # random-token repair: boost an irrelevant token by the same magnitude
        vocab = model.lm_head.weight.shape[0]
        rand_tok = np.random.randint(0, vocab)
        while rand_tok == gold_token:
            rand_tok = np.random.randint(0, vocab)
        rand_w = unembed_f[rand_tok].to(device)
        rand_vec = (max(gap, 0.0) / (rand_w.norm().item() ** 2 + 1e-8)) * rand_w
        g_rand = generate_with(make_repair_hook(rand_vec))
        rand_ctrl[group] += int(is_answer_correct(g_rand, aliases))

        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "group": group, "best_rank": best_rank,
            "peak_gold_logit": peak_gold_logit, "final_gold_logit": final_gold_logit,
            "gap": gap, "recovered": ok,
            "random_recover": bool(is_answer_correct(g_rand, aliases)),
        })

    print("\n" + "=" * 74)
    print("Decay repair guided by trajectory (restore gold logit to its peak)")
    print("=" * 74)
    for g in groups:
        if tot[g] == 0:
            print(f"\n[{g}] no samples")
            continue
        print(f"\n[{g}]  n={tot[g]} (best-rank <= k={args.k})")
        print(f"  restore gold logit to peak: {rec[g]}/{tot[g]} ({100*rec[g]/tot[g]:.1f}%)")
        print(f"  random-token repair:        {rand_ctrl[g]}/{tot[g]} ({100*rand_ctrl[g]/tot[g]:.1f}%)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"k": args.k, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
