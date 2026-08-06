"""
Gentle logit repair guided by the trajectory (Prediction -> Intervention).

Measured-decay-repair (setting the gold logit to its absolute intermediate
peak) failed because error samples' gold logit is negative while the current
top-1 logit is positive; an absolute peak never reaches the top, so the first
token never changed.

Here we repair gently and RELATIVELY: set the gold token's logit to
(current top-1 logit) + delta, with a SMALL delta. This is the minimal
intervention that makes the gold token competitive again. The informative
comparison is preservation vs formation for FULL-ANSWER recovery:
  - preservation: the representation did encode the answer (gold was
    competitive at an intermediate layer), so once gold is back on top the
    rest of the decoding is supported -> the full answer often recovers.
  - formation: the representation never encoded the answer; forcing gold on
    top only changes the first token and decoding then diverges -> full-answer
    recovery stays low.

So the trajectory predicts repairability: under the same gentle repair,
preservation should recover far more full answers than formation.

Controls: repair a random token (top + delta) -> no full-answer recovery.

Usage:
  python run_recovery.py --n 120 --k 5 --deltas 0.5,1.0,2.0
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
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
    parser.add_argument("--deltas", type=str, default="0.5,1.0,2.0")
    args = parser.parse_args()
    deltas = [float(x) for x in args.deltas.split(",")]
    max_delta = max(deltas)

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

    class TopPlusDeltaProcessor(LogitsProcessor):
        def __init__(self, token_id, delta):
            self.token_id = token_id
            self.delta = delta

        def __call__(self, input_ids, scores):
            scores = scores.clone()
            top_val = scores.max(dim=-1, keepdim=True).values
            scores[:, self.token_id] = top_val[:, 0] + self.delta
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
    rec_full = {g: {d: 0 for d in deltas} for g in groups}
    tot = {g: 0 for g in groups}
    rand_full = {g: 0 for g in groups}
    results = []

    for s in tqdm(errors[:args.n], desc="Gentle logit repair"):
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

        # classify preservation vs formation
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

        inter_ranks = []
        for l in range(num_layers - 1):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            inter_ranks.append((lr > lr[gold_token]).sum().item())
        best_rank = min(inter_ranks) if inter_ranks else 1e9
        group = "preservation" if best_rank <= args.k else "formation"
        tot[group] += 1

        def generate_forced(token_id, delta):
            procs = LogitsProcessorList([TopPlusDeltaProcessor(token_id, delta)])
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                 logits_processor=procs,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            return tokenizer.decode(out[0][input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()

        per_delta = {}
        for d in deltas:
            gen = generate_forced(gold_token, d)
            per_delta[d] = is_answer_correct(gen, aliases)
            rec_full[group][d] += int(per_delta[d])

        vocab = unembed_f.shape[0]
        rand_tok = np.random.randint(0, vocab)
        while rand_tok == gold_token:
            rand_tok = np.random.randint(0, vocab)
        g_rand = generate_forced(rand_tok, max_delta)
        rand_full[group] += int(is_answer_correct(g_rand, aliases))

        results.append({
            "id": s["id"], "task": s.get("task", "unknown"), "answer": s["answer"],
            "group": group, "best_rank": best_rank,
            "repair": {str(d): per_delta[d] for d in deltas},
            "random_repair": bool(is_answer_correct(g_rand, aliases)),
        })

    print("\n" + "=" * 74)
    print("Gentle logit repair (top+delta), preservation vs formation")
    print("=" * 74)
    for g in groups:
        if tot[g] == 0:
            print(f"\n[{g}] no samples")
            continue
        print(f"\n[{g}]  n={tot[g]} (best-rank <= k={args.k})")
        for d in deltas:
            print(f"  full-answer recovery (delta={d}): "
                  f"{rec_full[g][d]}/{tot[g]} ({100*rec_full[g][d]/tot[g]:.1f}%)")
        print(f"  random-token repair (delta={max_delta}): "
              f"{rand_full[g]}/{tot[g]} ({100*rand_full[g]/tot[g]:.1f}%)")

    out = os.path.join(config.DATA_DIR, "recovery_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({"deltas": deltas, "k": args.k, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
