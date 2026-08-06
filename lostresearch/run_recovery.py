"""
Decay repair via logit patching (Prediction -> Intervention).

Prior versions added a vector to the residual stream; the final RMSNorm
renormalization diluted it to ~zero, so nothing changed. This version patches
the logits directly: at generation time we hold the gold token's logit at its
own measured intermediate peak value. This is a measured repair (undo the
decay, hold the signal where it once was), NOT forcing gold to the absolute
top.

Prediction step: split errors by the trajectory:
  - preservation: gold was competitive (rank<=k) at some intermediate layer;
    its peak logit was high -> holding the final logit at the peak is a
    targeted repair of the decay.
  - formation: gold never competitive; its peak logit was low -> holding the
    final logit at the peak leaves it below the top, so repair does not help.

Headline: same repair, preservation recovers much more than formation. That is
the trajectory predicting repairability.

Implementation of "hold logit at peak": during generation, after lm_head,
set logits[gold] = peak_gold_logit (measured pre-run with the logit lens). We
do NOT touch other tokens, so the gold token rises to wherever its peak sat.

Control: patch a random token's logit to its own "peak" (same procedure) -> no
recovery expected.

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
    lm_head = model.lm_head

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

    for s in tqdm(errors[:args.n], desc="Decay repair (logit patch)"):
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

        # measure per-layer gold logit (logit lens) to find the peak and classify
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
        l_peak = int(np.argmax(gold_logits[:-1])) if len(gold_logits) > 1 else 0
        peak_gold_logit = gold_logits[l_peak]

        group = "preservation" if best_rank <= args.k else "formation"
        tot[group] += 1

        # ---- logit patching: hold gold logit at its peak during generation ----
        def make_logit_patch(patch_token, target_logit):
            state = {"active": True}
            orig_forward = None

            def patch_hook(module, args, output):
                if not state["active"]:
                    return output
                logits = output[0] if isinstance(output, tuple) else output
                # copy so we don't mutate in place across calls
                logits = logits.clone()
                logits[0, -1, patch_token] = target_logit
                if isinstance(output, tuple):
                    return (logits,) + output[1:]
                return logits

            hook = lm_head.register_forward_hook(patch_hook)
            return hook, state

        hook_gold, state_gold = make_logit_patch(gold_token, peak_gold_logit)
        out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        hook_gold.remove()
        gen_gold = tokenizer.decode(out[0][input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
        ok = is_answer_correct(gen_gold, aliases)
        rec[group] += int(ok)

        # ---- control: patch a random token to its own peak logit ----
        vocab = unembed_f.shape[0]
        rand_tok = np.random.randint(0, vocab)
        while rand_tok == gold_token:
            rand_tok = np.random.randint(0, vocab)
        # measure random token's peak logit the same way
        rand_peak = -1e9
        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            rand_peak = max(rand_peak, lr[rand_tok].item())
        hook_rand, _ = make_logit_patch(rand_tok, rand_peak)
        out_r = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        hook_rand.remove()
        gen_rand = tokenizer.decode(out_r[0][input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
        rand_ctrl[group] += int(is_answer_correct(gen_rand, aliases))

        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "group": group, "best_rank": best_rank,
            "peak_gold_logit": peak_gold_logit, "final_gold_logit": gold_logits[-1],
            "recovered": ok,
            "random_recover": bool(is_answer_correct(gen_rand, aliases)),
        })

    print("\n" + "=" * 74)
    print("Logit-patch decay repair guided by trajectory")
    print("=" * 74)
    for g in groups:
        if tot[g] == 0:
            print(f"\n[{g}] no samples")
            continue
        print(f"\n[{g}]  n={tot[g]} (best-rank <= k={args.k})")
        print(f"  hold gold logit at its peak: {rec[g]}/{tot[g]} ({100*rec[g]/tot[g]:.1f}%)")
        print(f"  random-token patch control:  {rand_ctrl[g]}/{tot[g]} ({100*rand_ctrl[g]/tot[g]:.1f}%)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"k": args.k, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
