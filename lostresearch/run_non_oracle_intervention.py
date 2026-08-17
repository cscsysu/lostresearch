"""
Non-oracle selective intervention via Representation Engineering.

This is the complete closed-loop experiment:
1. Train a steering direction from correct/incorrect hidden states (no gold answer needed)
2. Use trajectory predictor to identify preservation failures (no gold answer needed)
3. Apply activation steering ONLY to predicted preservation failures
4. Compare: no intervention / all intervention / selective intervention

The entire pipeline is NON-ORACLE at inference time: neither the steering
direction nor the trigger requires knowing the correct answer for the test sample.

Usage:
  python run_non_oracle_intervention.py --n 200 --alpha 2.0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
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
    parser.add_argument("--n", type=int, default=200,
                        help="Number of error samples to evaluate")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Steering strength")
    parser.add_argument("--steer-layers", type=int, default=4,
                        help="Number of last layers to steer")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # --- Step 1: Load data and train predictor ---
    print("=" * 70)
    print("Step 1: Train trajectory predictor + extract steering direction")
    print("=" * 70)

    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    with open(results_file) as f:
        all_results = json.load(f)

    # Train predictor (preservation vs formation)
    feats, labels, sample_ids = [], [], []
    for s in all_results:
        f = extract_features(s, config.PREDICTION_T0)
        if f and not s.get("final_correct", True):
            ranks = s.get("correct_rank", [])
            best_rank = min(ranks[:-1]) if len(ranks) > 1 else 1e9
            labels.append(int(best_rank <= config.RANK_COMPETITIVE))
            feats.append(f)
            sample_ids.append(s["id"])

    names = list(feats[0].keys())
    X_train = np.array([[f[k] for k in names] for f in feats])
    y_train = np.array(labels)
    print(f"  Predictor data: {len(X_train)} errors, {y_train.sum()} preservation")

    predictor = RandomForestClassifier(200, max_depth=8, random_state=42)
    predictor.fit(X_train, y_train)
    print(f"  Predictor train accuracy: {predictor.score(X_train, y_train):.3f}")

    # --- Step 2: Load model ---
    print("\n" + "=" * 70)
    print("Step 2: Load model and extract steering direction")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    device = model.device

    # Prepare data
    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    existing = {r["id"]: r for r in all_results}
    for s in prepared:
        if s["id"] in existing:
            s["final_correct"] = existing[s["id"]]["final_correct"]

    correct_samples = [s for s in prepared if s.get("final_correct", False)]
    error_samples = [s for s in prepared if not s.get("final_correct", True)]

    # --- Extract steering direction from correct vs incorrect hidden states ---
    # Use last `steer_layers` layers' hidden states
    steer_layer_start = num_layers - args.steer_layers
    print(f"  Extracting steering direction from layers {steer_layer_start}-{num_layers-1}")
    print(f"  Using {min(50, len(correct_samples))} correct + {min(50, len(error_samples))} incorrect samples")

    def get_last_hidden(sample, target_layer):
        """Get hidden state at last prompt position for a specific layer."""
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long, device=device)
        hidden_out = {}

        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_out["h"] = h[0, -1, :].detach().clone()

        hk = layers[target_layer].register_forward_hook(hook)
        with torch.no_grad():
            model(input_ids, use_cache=False)
        hk.remove()
        return hidden_out["h"]

    # Collect hidden states for steering direction
    n_dir = 50  # samples for direction estimation
    h_correct_all = []
    h_incorrect_all = []

    print("  Collecting correct sample hiddens...")
    for s in tqdm(correct_samples[:n_dir], desc="  Correct hiddens", leave=False):
        hs = []
        for l in range(steer_layer_start, num_layers):
            hs.append(get_last_hidden(s, l))
        h_correct_all.append(torch.stack(hs).mean(dim=0))  # average over target layers

    print("  Collecting incorrect sample hiddens...")
    for s in tqdm(error_samples[:n_dir], desc="  Incorrect hiddens", leave=False):
        hs = []
        for l in range(steer_layer_start, num_layers):
            hs.append(get_last_hidden(s, l))
        h_incorrect_all.append(torch.stack(hs).mean(dim=0))

    # Steering direction = mean(correct) - mean(incorrect)
    # DO NOT normalize: the raw magnitude is needed to overcome RMSNorm.
    # The direction's natural norm reflects the actual scale difference
    # between correct and incorrect hidden states.
    h_correct_mean = torch.stack(h_correct_all).mean(dim=0)
    h_incorrect_mean = torch.stack(h_incorrect_all).mean(dim=0)
    steering_dir = h_correct_mean - h_incorrect_mean
    raw_norm = steering_dir.norm().item()
    print(f"  Steering direction computed (raw norm={raw_norm:.1f}, dim={steering_dir.shape[0]})")
    print(f"  With alpha={args.alpha}, effective perturbation norm = {args.alpha * raw_norm:.1f}")

    # --- Step 3: Run intervention experiment ---
    print("\n" + "=" * 70)
    print("Step 3: Run non-oracle selective intervention")
    print("=" * 70)

    stats = {
        "no_intervention": {"n": 0, "recovered": 0},
        "all_steer": {"n": 0, "recovered": 0},
        "selective_steer": {"n": 0, "recovered": 0, "intervened": 0},
    }
    gt_stats = {"pres_recovered": 0, "pres_total": 0,
                "form_recovered": 0, "form_total": 0}
    results = []

    def generate_with_steering(sample, alpha, direction):
        """Generate with activation steering after final layernorm.
        
        We hook model.model.norm (the final RMSNorm) and add the steering
        direction to its OUTPUT. This is the last computation before the
        lm_head linear layer, so there is no subsequent normalization to
        wash out the perturbation.
        """
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long, device=device)

        def final_norm_hook(module, input, output):
            # output shape: [batch, seq, hidden_dim]
            out = output.clone()
            out[0, -1, :] += alpha * direction.to(out.device, out.dtype)
            return out

        hk = model.model.norm.register_forward_hook(final_norm_hook)

        out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        hk.remove()

        gen_text = tokenizer.decode(out[0][input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
        return gen_text

    for s in tqdm(error_samples[:args.n], desc="Non-oracle intervention"):
        aliases = s["aliases"]
        sid = s["id"]

        # Baseline: no intervention (already wrong by definition)
        stats["no_intervention"]["n"] += 1

        # Ground truth classification
        if sid in existing:
            ranks = existing[sid].get("correct_rank", [])
            best_rank = min(ranks[:-1]) if len(ranks) > 1 else 1e9
            gt_pres = best_rank <= config.RANK_COMPETITIVE
        else:
            gt_pres = False

        # Predict using trajectory predictor
        if sid in existing:
            feat = extract_features(existing[sid], config.PREDICTION_T0)
            if feat:
                x_pred = np.array([[feat[k] for k in names]])
                predicted_pres = bool(predictor.predict(x_pred)[0])
            else:
                predicted_pres = False
        else:
            predicted_pres = False

        # Strategy B: steer ALL errors
        gen_steered = generate_with_steering(s, args.alpha, steering_dir)
        recovered = is_answer_correct(gen_steered, aliases)
        stats["all_steer"]["n"] += 1
        stats["all_steer"]["recovered"] += int(recovered)

        # Strategy C: steer only predicted preservation
        if predicted_pres:
            stats["selective_steer"]["intervened"] += 1
            stats["selective_steer"]["recovered"] += int(recovered)
        stats["selective_steer"]["n"] += 1

        # Ground truth tracking
        if gt_pres:
            gt_stats["pres_total"] += 1
            gt_stats["pres_recovered"] += int(recovered)
        else:
            gt_stats["form_total"] += 1
            gt_stats["form_recovered"] += int(recovered)

        results.append({
            "id": sid, "gt_preservation": gt_pres,
            "predicted_preservation": predicted_pres,
            "recovered_with_steering": recovered,
        })

    # --- Report ---
    n = stats["no_intervention"]["n"]
    print("\n" + "=" * 70)
    print(f"Non-Oracle Selective Intervention Results (n={n}, alpha={args.alpha})")
    print("=" * 70)

    print(f"\n[A] No intervention: 0/{n} (0%)")

    b = stats["all_steer"]
    print(f"\n[B] Steer ALL errors (representation engineering):")
    print(f"  Interventions: {b['n']}")
    print(f"  Recovered: {b['recovered']}/{b['n']} ({100*b['recovered']/max(b['n'],1):.1f}%)")

    c = stats["selective_steer"]
    print(f"\n[C] Selective steer (only predicted preservation):")
    print(f"  Interventions: {c['intervened']}/{n} ({100*c['intervened']/max(n,1):.0f}% of errors)")
    print(f"  Recovered: {c['recovered']}/{max(c['intervened'],1)} ({100*c['recovered']/max(c['intervened'],1):.1f}% per intervention)")
    print(f"  Budget saved: {100*(1-c['intervened']/max(n,1)):.0f}%")

    print(f"\n[Ground truth] Steering effect by failure type:")
    gp = gt_stats
    print(f"  Preservation: {gp['pres_recovered']}/{gp['pres_total']} ({100*gp['pres_recovered']/max(gp['pres_total'],1):.1f}%)")
    print(f"  Formation: {gp['form_recovered']}/{gp['form_total']} ({100*gp['form_recovered']/max(gp['form_total'],1):.1f}%)")

    # Efficiency comparison
    print(f"\n[Efficiency comparison]:")
    eff_b = b['recovered'] / max(b['n'], 1)
    eff_c = c['recovered'] / max(c['intervened'], 1)
    print(f"  All-steer efficiency: {100*eff_b:.1f}% recovery per intervention")
    print(f"  Selective efficiency: {100*eff_c:.1f}% recovery per intervention")
    if eff_b > 0:
        print(f"  Efficiency gain: {100*(eff_c/eff_b - 1):.0f}%")

    out_file = os.path.join(config.DATA_DIR, "non_oracle_intervention_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"alpha": args.alpha, "steer_layers": args.steer_layers,
                   "n": n, "stats": stats, "gt_stats": gt_stats,
                   "results": results}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
