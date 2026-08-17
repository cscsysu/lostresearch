"""
Non-oracle TRIGGERED intervention v2 (task-balanced redesign).

v1 design flaws (why the 11% run was unrepresentative):
  1. Eval set = first-N errors in file order -> ALL TriviaQA. The trigger's
     ranking was evaluated on a single task.
  2. Trigger scores from a model pooled across 7 tasks, but evaluated on one
     task -- the wrong deployment configuration given task-sensitive features.
  3. Steering direction estimated from the first 50 correct/incorrect samples,
     also TriviaQA-dominated.

v2 fixes:
  1. Task-balanced eval: up to --per-task errors from EACH of the 7 tasks.
  2. Per-task trigger calibration: within-task 5-fold OOF probabilities
     (deployment story: calibrate on same-task labeled errors). Pooled trigger
     reported for comparison.
  3. Steering direction stratified across tasks.
  4. Coverage sweep {10,20,30,50}% from a single steered pass per sample.

Usage:
  python run_nonoracle_trigger_intervention.py --per-task 30 --alpha 1.0
"""
import argparse
import json
import os
import sys
from collections import defaultdict

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
    parser.add_argument("--per-task", type=int, default=30,
                        help="Errors per task in the balanced eval set")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--coverages", type=str, default="0.1,0.2,0.3,0.5")
    parser.add_argument("--steer-layers", type=int, default=4)
    args = parser.parse_args()
    coverages = [float(c) for c in args.coverages.split(",")]

    from transformers import AutoTokenizer, AutoModelForCausalLM

    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        alt = os.path.join(os.path.dirname(config.BASE_DIR), "lost-output",
                           "outputs", "data", "full_results_Qwen3-8B.json")
        results_file = alt if os.path.exists(alt) else results_file
    with open(results_file) as f:
        all_results = json.load(f)
    existing = {r["id"]: r for r in all_results}

    # --- Step 1: features + labels for ALL errors ---
    errors, feats, labels = [], [], []
    for s in all_results:
        if s.get("final_correct", True):
            continue
        fe = inference_available_features(s)
        ranks = s.get("correct_rank", [])
        if fe is None or len(ranks) < 2:
            continue
        best = min(ranks[:-1]) if len(ranks) > 1 else 1e9
        errors.append(s)
        feats.append(fe)
        labels.append(int(best <= config.RANK_COMPETITIVE))

    names = list(feats[0].keys())
    X_all = np.array([[f[k] for k in names] for f in feats])
    y_all = np.array(labels)
    task_all = np.array([s.get("task", "?") for s in errors])
    id_all = [s["id"] for s in errors]
    print(f"Errors with features: {len(X_all)}, positives {y_all.sum()}")

    # --- Step 2: trigger scores ---
    # (a) per-task calibration: within-task 5-fold OOF (deployment config)
    oof_percask = np.zeros(len(X_all))
    for t in np.unique(task_all):
        m = task_all == t
        if m.sum() < 20 or len(np.unique(y_all[m])) < 2:
            # too small to calibrate: fall back to pooled score later
            continue
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(500, max_depth=10, random_state=42)
        oof_percask[m] = cross_val_predict(rf, X_all[m], y_all[m], cv=skf,
                                           method="predict_proba")[:, 1]
    # (b) pooled calibration (v1 config, for comparison)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(500, max_depth=10, random_state=42)
    oof_pooled = cross_val_predict(rf, X_all, y_all, cv=skf,
                                   method="predict_proba")[:, 1]
    score_map_pertask = {i: float(p) for i, p in zip(id_all, oof_percask)}
    score_map_pooled = {i: float(p) for i, p in zip(id_all, oof_pooled)}
    print("Trigger scores ready (per-task + pooled).")

    # --- Step 3: task-balanced eval set ---
    by_task = defaultdict(list)
    for i, s in enumerate(errors):
        by_task[s.get("task", "?")].append(i)
    eval_idx = []
    for t, idxs in sorted(by_task.items()):
        eval_idx.extend(idxs[:args.per_task])
    print(f"\nBalanced eval set: {len(eval_idx)} errors "
          f"({', '.join(f'{t}:{min(len(v), args.per_task)}' for t, v in sorted(by_task.items()))})")

    # --- Step 4: model + task-stratified steering direction ---
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
    prep_map = {s["id"]: s for s in prepared}

    # stratified: up to 10 correct + 10 incorrect per task
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

    corr_by_task = defaultdict(list)
    err_by_task = defaultdict(list)
    for s in prepared:
        t = s.get("task", existing.get(s["id"], {}).get("task", "?"))
        if s["id"] not in existing:
            continue
        (corr_by_task if s.get("final_correct") else err_by_task)[t].append(s)

    h_c, h_i = [], []
    n_per_task = 10
    for t in sorted(set(list(corr_by_task) + list(err_by_task))):
        for s in corr_by_task[t][:n_per_task]:
            hs = [get_last_hidden(s, l) for l in range(steer_layer_start, num_layers)]
            h_c.append(torch.stack(hs).mean(dim=0))
        for s in err_by_task[t][:n_per_task]:
            hs = [get_last_hidden(s, l) for l in range(steer_layer_start, num_layers)]
            h_i.append(torch.stack(hs).mean(dim=0))
    print(f"  Steering direction from {len(h_c)} correct / {len(h_i)} incorrect "
          f"(task-stratified, {n_per_task}/task)")
    steering_dir = torch.stack(h_c).mean(dim=0) - torch.stack(h_i).mean(dim=0)
    print(f"  Direction norm = {steering_dir.norm().item():.1f}")

    # --- Step 5: one steered pass per eval sample; sweep coverages ---
    def generate_with_steering(sample):
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long, device=device)

        def final_norm_hook(module, input, output):
            out = output.clone()
            out[0, -1, :] += args.alpha * steering_dir.to(out.device, out.dtype)
            return out

        hk = model.model.norm.register_forward_hook(final_norm_hook)
        out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        hk.remove()
        return tokenizer.decode(out[0][input_ids.shape[1]:],
                                skip_special_tokens=True).strip()

    recs = []
    for i in tqdm(eval_idx, desc="Balanced triggered intervention"):
        s = errors[i]
        prep = prep_map.get(s["id"])
        if prep is None:
            continue
        gen = generate_with_steering(prep)
        rec = is_answer_correct(gen, prep["aliases"])
        ranks = s.get("correct_rank", [])
        best = min(ranks[:-1]) if len(ranks) > 1 else 1e9
        recs.append({"id": s["id"], "task": s.get("task", "?"),
                     "gt_pres": best <= config.RANK_COMPETITIVE,
                     "rec": int(rec),
                     "p_pertask": score_map_pertask[s["id"]],
                     "p_pooled": score_map_pooled[s["id"]]})

    n = len(recs)
    all_rec = sum(r["rec"] for r in recs)
    gt_pres_n = sum(1 for r in recs if r["gt_pres"])
    gt_pres_rec = sum(r["rec"] for r in recs if r["gt_pres"])
    gt_form_rec = sum(r["rec"] for r in recs if not r["gt_pres"])
    print("\n" + "=" * 70)
    print(f"Balanced non-oracle triggered intervention (n={n}, alpha={args.alpha})")
    print("=" * 70)
    print(f"[B] Steer ALL: {all_rec}/{n} ({100*all_rec/n:.1f}% per intervention)")
    print(f"[GT] preservation {gt_pres_rec}/{gt_pres_n} "
          f"({100*gt_pres_rec/max(gt_pres_n,1):.1f}%) vs formation "
          f"{gt_form_rec}/{n-gt_pres_n} ({100*gt_form_rec/max(n-gt_pres_n,1):.1f}%)")

    for trig_key, trig_name in [("p_pertask", "per-task"), ("p_pooled", "pooled")]:
        print(f"\n[C] Trigger = {trig_name} calibration:")
        scores = np.array([r[trig_key] for r in recs])
        recs_sorted = sorted(recs, key=lambda r: -r[trig_key])
        for cov in coverages:
            k = max(1, int(round(n * cov)))
            sel = recs_sorted[:k]
            rec_sel = sum(r["rec"] for r in sel)
            pres_sel = sum(1 for r in sel if r["gt_pres"])
            eff = rec_sel / k
            gain = eff / (all_rec / n) - 1 if all_rec > 0 else 0
            print(f"  top {100*cov:2.0f}%: intervened {k}, recovered {rec_sel} "
                  f"({100*eff:.1f}%/interv., gain {100*gain:+.0f}%), "
                  f"pres-precision {pres_sel}/{k} ({100*pres_sel/k:.0f}%)")

    out_file = os.path.join(config.DATA_DIR, "nonoracle_trigger_intervention_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"alpha": args.alpha, "per_task": args.per_task,
                   "coverages": coverages, "n": n,
                   "steer_all": {"rec": all_rec, "n": n},
                   "gt": {"pres_rec": gt_pres_rec, "pres_n": gt_pres_n,
                          "form_rec": gt_form_rec, "form_n": n - gt_pres_n},
                   "records": recs}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
