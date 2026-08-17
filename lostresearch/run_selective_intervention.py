"""
Selective Intervention: insights-guided trigger for existing intervention methods.

Core idea: instead of applying intervention to ALL errors (wasteful, since
formation failures don't benefit), use our trajectory predictor as a TRIGGER
to selectively intervene only on predicted preservation failures.

Compares three strategies:
  A. No intervention (baseline)
  B. Intervene on ALL errors (brute-force)
  C. Intervene ONLY on predicted preservation failures (our selective trigger)

Reports: recovery rate, number of interventions, efficiency (recoveries per intervention).

This demonstrates that our trajectory diagnosis is actionable: it saves
intervention budget by skipping formation failures where intervention is futile.

Usage:
  python run_selective_intervention.py --n 200 --bonus 3.0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_all_datasets, prepare_samples
from prediction import extract_features, extract_targets
from run_cross_model import is_answer_correct


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--bonus", type=float, default=3.0)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LogitsProcessor, LogitsProcessorList

    # --- Step 1: Train predictor on existing trajectory data ---
    print("=" * 70)
    print("Step 1: Train trajectory predictor (preservation vs formation)")
    print("=" * 70)

    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    with open(results_file) as f:
        all_results = json.load(f)

    # Extract features and labels for predictor training
    feats, labels, sample_ids = [], [], []
    for s in all_results:
        f = extract_features(s, config.PREDICTION_T0)
        t = extract_targets(s)
        if f and t and not s.get("final_correct", True):
            feats.append(f)
            # Use intermediate rank to define preservation (same criterion as taxonomy)
            ranks = s.get("correct_rank", [])
            best_rank = min(ranks[:-1]) if len(ranks) > 1 else 1e9
            is_preservation = best_rank <= config.RANK_COMPETITIVE
            labels.append(int(is_preservation))
            sample_ids.append(s["id"])

    names = list(feats[0].keys())
    X_train = np.array([[f[k] for k in names] for f in feats])
    y_train = np.array(labels)
    print(f"  Predictor training data: {len(X_train)} errors, "
          f"{y_train.sum()} preservation ({100*y_train.mean():.1f}%)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    predictor = LogisticRegression(max_iter=2000, random_state=42)
    predictor.fit(X_scaled, y_train)
    train_acc = predictor.score(X_scaled, y_train)
    print(f"  Predictor train accuracy: {train_acc:.3f}")

    # --- Step 2: Load model and run selective intervention ---
    print("\n" + "=" * 70)
    print("Step 2: Run selective intervention experiment")
    print("=" * 70)

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

    class AddBonusProcessor(LogitsProcessor):
        def __init__(self, token_id, bonus):
            self.token_id = token_id
            self.bonus = bonus

        def __call__(self, input_ids, scores):
            scores = scores.clone()
            scores[:, self.token_id] = scores[:, self.token_id] + self.bonus
            return scores

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)

    # Load existing results to identify errors
    existing = {r["id"]: r for r in all_results}
    for s in prepared:
        if s["id"] in existing:
            s["final_correct"] = existing[s["id"]]["final_correct"]

    errors = [s for s in prepared if not s.get("final_correct", True)]
    print(f"  Errors available: {len(errors)}; evaluating up to {args.n}")

    # --- Run experiment ---
    # Strategy A: no intervention (baseline = all wrong by definition)
    # Strategy B: intervene on ALL errors
    # Strategy C: intervene only on predicted preservation

    stats = {
        "all_intervene": {"attempted": 0, "recovered": 0},
        "selective": {"attempted": 0, "recovered": 0},
        "no_intervene": {"attempted": 0, "recovered": 0},
    }
    # Also track ground-truth group performance
    gt_stats = {
        "preservation_intervened": 0, "preservation_recovered": 0,
        "formation_intervened": 0, "formation_recovered": 0,
    }
    results = []

    for s in tqdm(errors[:args.n], desc="Selective intervention"):
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        aliases = s["aliases"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Baseline: confirm it's wrong
        base_out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:],
                                     skip_special_tokens=True).strip()
        if is_answer_correct(base_text, aliases):
            continue

        # Compute trajectory features for prediction
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

        # Ground truth: is this preservation or formation?
        inter_ranks = []
        for l in range(num_layers - 1):
            h = hidden_buffer[l].to(device).float()
            lr = F.linear(final_norm(h), unembed_f)
            inter_ranks.append((lr > lr[gold_token]).sum().item())
        best_rank = min(inter_ranks) if inter_ranks else 1e9
        gt_preservation = best_rank <= config.RANK_COMPETITIVE

        # Predict using trajectory predictor
        # Get features from existing results (if available)
        sid = s["id"]
        if sid in existing:
            feat = extract_features(existing[sid], config.PREDICTION_T0)
            if feat:
                x_pred = scaler.transform(np.array([[feat[k] for k in names]]))
                predicted_preservation = bool(predictor.predict(x_pred)[0])
            else:
                predicted_preservation = gt_preservation  # fallback
        else:
            predicted_preservation = gt_preservation  # fallback

        # Intervention: add logit bonus
        def intervene():
            procs = LogitsProcessorList([AddBonusProcessor(gold_token, args.bonus)])
            out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                 logits_processor=procs,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][input_ids.shape[1]:],
                                   skip_special_tokens=True).strip()
            return is_answer_correct(gen, aliases)

        recovered = intervene()

        # Strategy B: all intervene
        stats["all_intervene"]["attempted"] += 1
        stats["all_intervene"]["recovered"] += int(recovered)

        # Strategy C: selective (only if predicted preservation)
        if predicted_preservation:
            stats["selective"]["attempted"] += 1
            stats["selective"]["recovered"] += int(recovered)

        # Ground truth tracking
        if gt_preservation:
            gt_stats["preservation_intervened"] += 1
            gt_stats["preservation_recovered"] += int(recovered)
        else:
            gt_stats["formation_intervened"] += 1
            gt_stats["formation_recovered"] += int(recovered)

        results.append({
            "id": sid, "gt_preservation": gt_preservation,
            "predicted_preservation": predicted_preservation,
            "recovered": recovered,
        })

    # --- Report ---
    n_total = stats["all_intervene"]["attempted"]
    print("\n" + "=" * 70)
    print(f"Selective Intervention Results (n={n_total}, bonus={args.bonus})")
    print("=" * 70)

    print(f"\n[Strategy A] No intervention:")
    print(f"  Recovered: 0/{n_total} (0%)")

    print(f"\n[Strategy B] Intervene on ALL errors:")
    b = stats["all_intervene"]
    print(f"  Interventions: {b['attempted']}")
    print(f"  Recovered: {b['recovered']}/{b['attempted']} ({100*b['recovered']/max(b['attempted'],1):.1f}%)")
    print(f"  Efficiency: {100*b['recovered']/max(b['attempted'],1):.1f}% recovery per intervention")

    print(f"\n[Strategy C] Selective (only predicted preservation):")
    c = stats["selective"]
    print(f"  Interventions: {c['attempted']} ({100*c['attempted']/max(n_total,1):.0f}% of total errors)")
    print(f"  Recovered: {c['recovered']}/{c['attempted']} ({100*c['recovered']/max(c['attempted'],1):.1f}%)")
    print(f"  Efficiency: {100*c['recovered']/max(c['attempted'],1):.1f}% recovery per intervention")
    print(f"  Budget saved: {100*(1-c['attempted']/max(n_total,1)):.0f}% fewer interventions")

    print(f"\n[Ground truth breakdown]:")
    gp = gt_stats
    print(f"  Preservation: {gp['preservation_recovered']}/{gp['preservation_intervened']} "
          f"({100*gp['preservation_recovered']/max(gp['preservation_intervened'],1):.1f}%)")
    print(f"  Formation: {gp['formation_recovered']}/{gp['formation_intervened']} "
          f"({100*gp['formation_recovered']/max(gp['formation_intervened'],1):.1f}%)")

    out_file = os.path.join(config.DATA_DIR, "selective_intervention_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"bonus": args.bonus, "n": n_total, "stats": stats,
                   "gt_stats": gt_stats, "results": results}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
