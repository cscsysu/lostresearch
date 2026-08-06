"""
Probe robustness: does the peak-rank separation survive a change of decoder?

Reviewer concern: the information trajectory is built on the tuned lens. The
headline claim is that correct examples reach a much better intermediate gold
PEAK RANK than incorrect ones (median ~0-2 for correct vs ~19-77 for incorrect).
Does that separation survive if we swap the decoder?

We decode each intermediate hidden state h_l and compute the gold token's rank
under three independent decoders:
  (a) raw logit lens :  softmax over W_U(fn(h))           (no training)
  (b) tuned lens     :  softmax over W_U(fn(A h + b))     (affine translators)
  (c) cosine         :  rank of gold by cos(h, w_y) across vocab  (no training)

For each decoder we report the median intermediate peak rank (excluding the
final layer) for correct vs incorrect examples, plus a Mann-Whitney U test.
If all three decoders show correct << incorrect, the conclusion is not an
artifact of a particular probe. Cosine is a strong control: it uses only the
direction of each token's unembedding vector and neither a trained map nor the
final norm.

NOTE: rank over the FULL vocabulary for every layer under every decoder is
expensive. We compute the gold rank exactly; this requires the dot product of h
with all unembedding vectors, which is one big matrix multiply per layer.

Usage:
  python run_probe_robustness.py [--n 200]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

import config
from data_loader import load_all_datasets, prepare_samples


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
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
                s["generated"] = existing[s["id"]].get("generated", "")

    translator_file = os.path.join(config.DATA_DIR, "tuned_lens_translators_Qwen3-8B.pt")
    translators = None
    if os.path.exists(translator_file):
        translators = torch.load(translator_file, map_location="cpu")
        print(f"Loaded translators from {translator_file}")
    else:
        print("No saved translators -> raw + cosine only.")

    decoders = ["raw", "cosine"]
    if translators is not None:
        decoders.append("tuned")

    # {decoder: {correct: [peak_rank], incorrect: [peak_rank]}}
    peaks = {d: {"correct": [], "incorrect": []} for d in decoders}

    for s in tqdm(prepared[:args.n], desc="Probe robustness"):
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        correct = s.get("final_correct", False)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
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

        # per-decoder per-layer gold rank (exclude final layer for peak)
        ranks = {d: [] for d in decoders}
        for l in range(num_layers - 1):  # exclude final layer
            h = hidden_buffer[l].to(device).float()

            # raw
            lr = F.linear(final_norm(h), unembed_f)
            ranks["raw"].append((lr > lr[gold_token]).sum().item())

            # tuned
            if translators is not None:
                A = translators[l]["A"].to(device).float()
                b = translators[l]["b"].to(device).float()
                lt = F.linear(final_norm(A @ h + b), unembed_f)
                ranks["tuned"].append((lt > lt[gold_token]).sum().item())

            # cosine: rank gold by cos(h, w_y) over full vocab
            cos_all = F.cosine_similarity(h.unsqueeze(0), unembed_f)  # [vocab]
            ranks["cosine"].append((cos_all > cos_all[gold_token]).sum().item())

        key = "correct" if correct else "incorrect"
        for d in decoders:
            if ranks[d]:
                peaks[d][key].append(min(ranks[d]))

    print("\n" + "=" * 70)
    print("Probe robustness: intermediate gold peak rank (excl. final), "
          "correct vs incorrect")
    print("=" * 70)
    for d in decoders:
        c = np.array(peaks[d]["correct"])
        i = np.array(peaks[d]["incorrect"])
        if len(c) == 0 or len(i) == 0:
            print(f"\n[{d}] insufficient (correct={len(c)}, incorrect={len(i)})")
            continue
        med_c, med_i = np.median(c), np.median(i)
        try:
            u, p = stats.mannwhitneyu(c, i, alternative="less")  # correct rank < incorrect
            sig = "significant" if p < 0.05 else "n.s."
        except Exception:
            u, p, sig = float("nan"), float("nan"), "?"
        print(f"\n[{d}]  correct n={len(c)}, incorrect n={len(i)}")
        print(f"  median peak rank: correct={med_c:.0f}, incorrect={med_i:.0f}")
        print(f"  Mann-Whitney (correct<incorrect): U={u:.0f}, p={p:.2e} -> {sig}")

    out = os.path.join(config.DATA_DIR, "probe_robustness_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({d: {k: v for k, v in peaks[d].items()} for d in decoders}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
