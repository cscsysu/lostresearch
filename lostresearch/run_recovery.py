"""
Measured decay repair via LogitsProcessor (Prediction -> Intervention).

We restore the gold token's logit to its OWN measured intermediate peak during
generation (a measured repair that undoes the decay), and evaluate at the
TOKEN level (does the first generated token become the gold token?) in addition
to full-answer correctness. Multi-token answers make full-answer recovery hard
even for a correct first token, so the token-level metric is the fair, direct
test of whether the gold signal was restored.

Prediction step (the Prediction -> Intervention link): split errors by the
trajectory into preservation (gold was competitive, peak logit high) vs
formation (gold never competitive, peak logit low). Under the SAME measured
repair (restore gold logit to its peak):
  - preservation: the peak was high enough to be competitive -> restoring it
    should make the first token gold (recovery);
  - formation: the peak was never high -> restoring it leaves gold below the
    top -> no recovery.

This is a measured repair (undo the decay), NOT forcing gold to the absolute
top: we set the gold logit to its historical peak, which for a formation
failure is still low.

Metrics reported per group:
  - first-token recovery: fraction where the generated first token == gold token
  - full-answer recovery: fraction where the whole generation matches an alias

Controls: restore a RANDOM token to its own peak (direction specificity).

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
    from transformers import LogitsProcessor, LogitsProcessorList
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

    class SetLogitProcessor(LogitsProcessor):
        def __init__(self, token_id, value):
            self.token_id = token_id
            self.value = value

        def __call__(self, input_ids, scores):
            scores = scores.clone()
            scores[:, self.token_id] = self.value
            return scores

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
    rec_first = {g: 0 for g in groups}
    rec_full = {g: 0 for g in groups}
    tot = {g: 0 for g in groups}
    ctrl_first = {g: 0 for g in groups}
    results = []

    for s in tqdm(errors[:args.n], desc="Measured decay repair"):
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        aliases = s["aliases"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        base_out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        base_ids = base_out[0][input_ids.shape[1]:].tolist()
        base_text = tokenizer.decode(base_ids, skip_special_tokens=True).strip()
        if is_answer_correct(base_text, aliases):
            continue

        # measure per-layer gold logit + rank; find peak (measured repair target)
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

        def generate_measured(token_id, value):
            procs = LogitsProcessorList([SetLogitProcessor(token_id, value)])
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                 logits_processor=procs,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            new_ids = out[0][input_ids.shape[1]:].tolist()
            return new_ids, tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # measured repair: hold gold logit at its intermediate peak
        new_ids, gen_text = generate_measured(gold_token, peak_gold_logit)
        first_ok = (len(new_ids) > 0) and (new_ids[0] == gold_token)
        full_ok = is_answer_correct(gen_text, aliases)
        rec_first[group] += int(first_ok)
        rec_full[group] += int(full_ok)

        # control: hold a RANDOM token at its own peak
        vocab = unembed_f.shape[0]
        rand_tok = np.random.randint(0, vocab)
        while rand_tok == gold_token:
            rand_tok = np.random.randint(0, vocab)
        rand_peak = -1e9
        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            rand_peak = max(rand_peak, lr[rand_tok].item())
        new_ids_r, _ = generate_measured(rand_tok, rand_peak)
        # does random repair make gold the first token? (should be no)
        ctrl_first[group] += int((len(new_ids_r) > 0) and (new_ids_r[0] == gold_token))

        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "group": group, "best_rank": best_rank,
            "peak_gold_logit": peak_gold_logit, "final_gold_logit": gold_logits[-1],
            "first_token_recovered": first_ok, "full_answer_recovered": full_ok,
            "base_first_token": base_ids[0] if base_ids else None,
            "repaired_first_token": new_ids[0] if new_ids else None,
        })

    print("\n" + "=" * 74)
    print("Measured decay repair (restore gold logit to its peak), token-level")
    print("=" * 74)
    for g in groups:
        if tot[g] == 0:
            print(f"\n[{g}] no samples")
            continue
        print(f"\n[{g}]  n={tot[g]} (best-rank <= k={args.k})")
        print(f"  first-token recovered:  {rec_first[g]}/{tot[g]} ({100*rec_first[g]/tot[g]:.1f}%)")
        print(f"  full-answer recovered:  {rec_full[g]}/{tot[g]} ({100*rec_full[g]/tot[g]:.1f}%)")
        print(f"  random-repair first-token->gold: {ctrl_first[g]}/{tot[g]} ({100*ctrl_first[g]/tot[g]:.1f}%)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"k": args.k, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
