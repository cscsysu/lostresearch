"""
Non-oracle TRIGGERED intervention: the trigger is the leak-audited
inference-available predictor (prefix top-5 features, no gold token).

Reviewer round-6 requirement: "drive an actual intervention with the
non-oracle predictor."

Pipeline:
 1. Load full_results; compute PREFIX-ONLY top-5 features for every error
    (same features as run_inference_available_predictor.py).
 2. Train RF; use OUT-OF-FOLD probabilities as the trigger score (no
    train-on-test leakage in the trigger).
 3. Steering direction: RepE contrast (correct vs incorrect last-layer
    hiddens), same as run_non_oracle_intervention.py.
 4. Conditions:
      [B] steer ALL errors
      [C] steer only errors whose oof-probability is in the top q coverage
    Compare recovery per intervention + ground-truth preservation/formation
    breakdown.

The ONLY difference from run_non_oracle_intervention.py is the trigger:
there the trigger used gold-conditioned trajectory features; here it uses
strictly prefix-only top-5 dynamics. Everything else is identical.

Usage:
  python run_nonoracle_trigger_intervention.py --n 200 --alpha 1.0 --coverage 0.3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_all_datasets, prepare_samples
from run_cross_model import is_answer_correct
from run_inference_available_predictor import inference_available_features


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Steering strength (1.0 was best in sweep)")
    parser.add_argument("--coverage", type=float, default=0.3,
                        help="Fraction of errors flagged by the trigger")
    parser.add_argument("--steer-layers", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as f:
        all_results = json.load(f)
    existing = {r["id"]: r for r in all_results}

    # --- Step 1: prefix-only features + out-of-fold trigger ---
    print("=" * 70)
    print("Step 1: inference-available trigger (out-of-fold, no leakage)")
    print("=" * 70)
    feats, labels, ids = [], [], []
    for s in all_results:
        if s.get("final_correct", True):
            continue
        fe = inference_available_features(s)
        ranks = s.get("correct_rank", [])
        if fe is None or len(ranks) < 2:
            continue
        best_rank = min(ranks[:-1]) if len(ranks) > 1 else 1e9
        feats.append(fe)
        labels.append(int(best_rank <= config.RANK_COMPETITIVE))
        ids.append(s["id"])

    names = list(feats[0].keys())
    X = np.array([[f[k] for k in names] for f in feats])
    y = np.array(labels)
    print(f"  {len(X)} errors, {y.sum()} preservation positives")

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(500, max_depth=10, random_state=42)
    oof = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
    oof_map = {i: float(p) for i, p in zip(ids, oof)}
    print(f"  Out-of-fold trigger probabilities computed for {len(ids)} errors")

    # --- Step 2: model + steering direction ---
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    device = model.device

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    for s in prepared:
        if s["id"] in existing:
            s["final_correct"] = existing[s["id"]]["final_correct"]
    correct_samples = [s for s in prepared if s.get("final_correct", False)]
    error_samples = [s for s in prepared if not s.get("final_correct", True)]

    steer_layer_start = num_layers - args.steer_layers

    def get_last_hidden(sample, target_layer):
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

    n_dir = 50
    h_c, h_i = [], []
    for s in tqdm(correct_samples[:n_dir], desc="Correct hiddens", leave=False):
        hs = [get_last_hidden(s, l) for l in range(steer_layer_start, num_layers)]
        h_c.append(torch.stack(hs).mean(dim=0))
    for s in tqdm(error_samples[:n_dir], desc="Incorrect hiddens", leave=False):
        hs = [get_last_hidden(s, l) for l in range(steer_layer_start, num_layers)]
        h_i.append(torch.stack(hs).mean(dim=0))

    steering_dir = torch.stack(h_c).mean(dim=0) - torch.stack(h_i).mean(dim=0)
    print(f"  Steering direction norm = {steering_dir.norm().item():.1f}")

    # --- Step 3: threshold from oof scores ---
    eval_errors = error_samples[:args.n]
    scores = [oof_map.get(s["id"], 0.0) for s in eval_errors]
    thresh = np.quantile(scores, 1 - args.coverage)
    print(f"\n  Trigger threshold (top {100*args.coverage:.0f}% coverage): p >= {thresh:.3f}")

    def generate_with_steering(sample, alpha, direction):
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long, device=device)

        def final_norm_hook(module, input, output):
            out = output.clone()
            out[0, -1, :] += alpha * direction.to(out.device, out.dtype)
            return out

        hk = model.model.norm.register_forward_hook(final_norm_hook)
        out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        hk.remove()
        return tokenizer.decode(out[0][input_ids.shape[1]:],
                                skip_special_tokens=True).strip()

    stats = {"all_steer": {"n": 0, "rec": 0},
             "selective": {"n": 0, "intervened": 0, "rec": 0}}
    gt = {"pres_rec": 0, "pres_n": 0, "form_rec": 0, "form_n": 0}
    flagged_correctly = 0

    for s in tqdm(eval_errors, desc="Triggered intervention"):
        sid = s["id"]
        gen = generate_with_steering(s, args.alpha, steering_dir)
        rec = is_answer_correct(gen, s["aliases"])

        stats["all_steer"]["n"] += 1
        stats["all_steer"]["rec"] += int(rec)

        p = oof_map.get(sid, 0.0)
        if p >= thresh:
            stats["selective"]["intervened"] += 1
            stats["selective"]["rec"] += int(rec)
        stats["selective"]["n"] += 1

        ranks = existing[sid].get("correct_rank", []) if sid in existing else []
        best = min(ranks[:-1]) if len(ranks) > 1 else 1e9
        if best <= config.RANK_COMPETITIVE:
            gt["pres_n"] += 1
            gt["pres_rec"] += int(rec)
            if p >= thresh:
                flagged_correctly += 1
        else:
            gt["form_n"] += 1
            gt["form_rec"] += int(rec)

    print("\n" + "=" * 70)
    print(f"Non-Oracle TRIGGERED intervention (n={stats['all_steer']['n']}, "
          f"alpha={args.alpha}, coverage={args.coverage})")
    print("=" * 70)
    b = stats["all_steer"]
    c = stats["selective"]
    print(f"\n[B] Steer ALL:  {b['rec']}/{b['n']} "
          f"({100*b['rec']/max(b['n'],1):.1f}% per intervention)")
    print(f"[C] TRIGGER-only (top {100*args.coverage:.0f}% by oof score): "
          f"{c['rec']}/{c['intervened']} "
          f"({100*c['rec']/max(c['intervened'],1):.1f}% per intervention)")
    print(f"    Budget saved: {100*(1 - c['intervened']/max(c['n'],1)):.0f}%")
    eff_b = b["rec"] / max(b["n"], 1)
    eff_c = c["rec"] / max(c["intervened"], 1)
    if eff_b > 0:
        print(f"    Efficiency gain: {100*(eff_c/eff_b - 1):.0f}%")
    print(f"\n[Ground truth under steering]")
    print(f"  Preservation: {gt['pres_rec']}/{gt['pres_n']} "
          f"({100*gt['pres_rec']/max(gt['pres_n'],1):.1f}%)")
    print(f"  Formation:    {gt['form_rec']}/{gt['form_n']} "
          f"({100*gt['form_rec']/max(gt['form_n'],1):.1f}%)")
    print(f"  Pres positives captured by trigger: {flagged_correctly}/{gt['pres_n']}")

    out_file = os.path.join(config.DATA_DIR, "nonoracle_trigger_intervention_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"alpha": args.alpha, "coverage": args.coverage,
                   "threshold": float(thresh), "stats": stats, "gt": gt}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
