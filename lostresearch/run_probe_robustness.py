"""
Probe robustness: does the correct/incorrect signal separation survive a
change of decoder?

Reviewer concern: the information trajectory is built on the tuned lens. The
headline claim is that correct examples build a stronger intermediate
gold-answer signal than incorrect ones. Does that separation survive if we swap
the decoder?

We decode each intermediate hidden state h_l into a scalar "gold support" under
three independent decoders:
  (a) raw logit lens :  final_norm(h) -> W_U[gold]       (no training)
  (b) tuned lens     :  final_norm(A h + b) -> W_U[gold] (affine translators)
  (c) cosine         :  cos(h, w_gold)                    (no training, no norm)

For each decoder we report, on the same samples:
  - median first-half gold-support peak for correct vs incorrect examples,
  - the separation, and a Mann-Whitney U test.

If the correct > incorrect separation holds under all three decoders, the
conclusion is not an artifact of a particular probe. Cosine is a strong control
because it uses only the direction of the gold unembedding and involves neither
a trained map nor the final norm.

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
        print("No saved translators -> raw + cosine only (tuned needs "
              "run_p0_tuned_lens.py first).")

    decoders = ["raw", "cosine"]
    if translators is not None:
        decoders.append("tuned")

    # {decoder: {correct: [peak], incorrect: [peak]}}
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

        gold_w = unembed_f[gold_token]
        mid = num_layers // 2
        mid_scores = {d: [] for d in decoders}
        for l in range(mid):
            h = hidden_buffer[l].to(device).float()
            mid_scores["raw"].append(F.linear(final_norm(h), unembed_f)[gold_token].item())
            mid_scores["cosine"].append(
                F.cosine_similarity(h.unsqueeze(0), gold_w.unsqueeze(0)).item())
            if translators is not None:
                A = translators[l]["A"].to(device).float()
                b = translators[l]["b"].to(device).float()
                lt = F.linear(final_norm(A @ h + b), unembed_f)
                mid_scores["tuned"].append(lt[gold_token].item())

        key = "correct" if correct else "incorrect"
        for d in decoders:
            peaks[d][key].append(max(mid_scores[d]) if mid_scores[d] else 0.0)

    print("\n" + "=" * 70)
    print("Probe robustness: first-half gold-support peak, correct vs incorrect")
    print("=" * 70)
    for d in decoders:
        c = np.array(peaks[d]["correct"])
        i = np.array(peaks[d]["incorrect"])
        if len(c) == 0 or len(i) == 0:
            print(f"\n[{d}] insufficient data (correct={len(c)}, incorrect={len(i)})")
            continue
        med_c, med_i = np.median(c), np.median(i)
        try:
            u, p = stats.mannwhitneyu(c, i, alternative="greater")
            sig = "significant" if p < 0.05 else "n.s."
        except Exception:
            u, p, sig = float("nan"), float("nan"), "?"
        print(f"\n[{d}]")
        print(f"  median first-half peak: correct={med_c:.3f}, incorrect={med_i:.3f}")
        print(f"  separation (correct-incorrect) = {med_c - med_i:+.3f}")
        print(f"  Mann-Whitney (correct>incorrect): U={u:.0f}, p={p:.2e} -> {sig}")

    out = os.path.join(config.DATA_DIR, "probe_robustness_Qwen3-8B.json")
    with open(out, "w") as f:
        json.dump({d: {k: v for k, v in peaks[d].items()} for d in decoders}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
