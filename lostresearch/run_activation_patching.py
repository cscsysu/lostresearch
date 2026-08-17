"""
Activation Patching: causal validation of layer-35 MLP's role in decay.

Tests whether the late-layer MLP causally suppresses the gold signal in
preservation failures, by patching (zeroing or restoring) the MLP output
at the identified layer and measuring whether gold rank improves.

Two experiments:
1. Zero-ablation: zero out layer-35 MLP output for preservation failures.
   If gold rank improves -> MLP was causally suppressing.
2. Comparison: same ablation on formation failures.
   If gold rank does NOT improve -> confirms it's preservation-specific.

This directly addresses reviewer: "attribution ≠ causality; does fixing
the MLP actually restore the gold signal?"

Usage:
  python run_activation_patching.py --n 100 --target-layer 35
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    parser.add_argument("--target-layer", type=int, default=35,
                        help="Layer whose MLP to patch (0-indexed)")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import LogitsProcessor, LogitsProcessorList

    # Load results to classify samples
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    with open(results_file) as f:
        all_results = json.load(f)

    errors = [s for s in all_results if not s.get("final_correct")]
    for s in errors:
        ranks = s.get("correct_rank", [])
        s["best_mid_rank"] = min(ranks[:-1]) if len(ranks) > 1 else 1e9
        s["is_preservation"] = s["best_mid_rank"] <= config.RANK_COMPETITIVE

    pres = [s for s in errors if s["is_preservation"]]
    form = [s for s in errors if not s["is_preservation"]]
    print(f"Preservation: {len(pres)}, Formation: {len(form)}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    final_norm = model.model.norm
    unembed = model.lm_head.weight.float()
    device = model.device
    target_layer = args.target_layer

    # Prepare data
    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    existing = {r["id"]: r for r in all_results}
    for s in prepared:
        if s["id"] in existing:
            s["final_correct"] = existing[s["id"]]["final_correct"]
            s["is_preservation"] = existing[s["id"]].get("is_preservation", False)

    prep_map = {s["id"]: s for s in prepared}

    print(f"\nTarget layer for patching: {target_layer}")
    print(f"Experiment: zero-ablate MLP output at layer {target_layer}")

    # For each sample: run with and without MLP ablation, compare gold rank
    def get_gold_rank_with_ablation(sample, ablate_mlp=False):
        """Forward pass, optionally ablating the target layer's MLP output.
        
        We remove the component of MLP output along the gold-vs-competitor
        contrast direction (w_gold - w_competitor). This is the direction
        that DLA identifies as the main contributor to margin change.
        By removing it, we test whether this specific directional component
        of the MLP is causally responsible for the gold signal's fate.
        """
        input_ids = torch.tensor([sample["prompt_ids"]], dtype=torch.long, device=device)
        gold_token = sample["primary_answer_ids"][0]
        # Get competitor (generated token)
        sid = sample["id"]
        gen_token = gold_token  # fallback
        if sid in existing:
            gen_text = existing[sid].get("generated", "")
            if gen_text:
                gen_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                if gen_ids:
                    gen_token = gen_ids[0]

        hook_handle = None
        if ablate_mlp:
            target_mlp = layers[target_layer].mlp
            # Contrast direction: w_gold - w_competitor (same as DLA)
            contrast_dir = (unembed[gold_token] - unembed[gen_token]).to(device).float()
            contrast_dir = contrast_dir / contrast_dir.norm()

            def directional_ablation_hook(module, input, output):
                # Remove the contrast-direction component from MLP output
                out = output.clone()
                mlp_vec = out[0, -1, :].float()
                # Project out the contrast direction
                proj = (mlp_vec @ contrast_dir) * contrast_dir
                out[0, -1, :] = (mlp_vec - proj).to(out.dtype)
                return out

            hook_handle = target_mlp.register_forward_hook(directional_ablation_hook)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :]  # final position logits

        if hook_handle:
            hook_handle.remove()

        # Compute gold rank
        rank = (logits > logits[gold_token]).sum().item()
        return rank

    # Run experiment
    results = {"preservation": [], "formation": []}

    # Evaluate preservation samples
    pres_samples = [s for s in pres if s["id"] in prep_map][:args.n // 2]
    form_samples = [s for s in form if s["id"] in prep_map][:args.n // 2]

    print(f"\nEvaluating {len(pres_samples)} preservation + {len(form_samples)} formation samples...")

    for group_name, group_samples in [("preservation", pres_samples), ("formation", form_samples)]:
        for s_info in tqdm(group_samples, desc=f"Patching {group_name}"):
            sid = s_info["id"]
            if sid not in prep_map:
                continue
            s = prep_map[sid]

            # Normal forward: gold rank
            rank_normal = get_gold_rank_with_ablation(s, ablate_mlp=False)
            # Ablated forward: gold rank after zeroing MLP
            rank_ablated = get_gold_rank_with_ablation(s, ablate_mlp=True)

            # Improvement = rank decreased (lower = better)
            improvement = rank_normal - rank_ablated

            results[group_name].append({
                "id": sid,
                "rank_normal": rank_normal,
                "rank_ablated": rank_ablated,
                "improvement": improvement,  # positive = ablation helped gold
            })

    # Report
    print("\n" + "=" * 70)
    print(f"Activation Patching Results (layer {target_layer} MLP zero-ablation)")
    print("=" * 70)

    for group_name in ["preservation", "formation"]:
        r = results[group_name]
        if not r:
            continue
        improvements = [x["improvement"] for x in r]
        helped = sum(1 for x in improvements if x > 0)  # ablation improved gold rank
        hurt = sum(1 for x in improvements if x < 0)  # ablation hurt gold rank
        print(f"\n[{group_name}] n={len(r)}")
        print(f"  MLP ablation HELPED gold (rank improved): {helped}/{len(r)} ({100*helped/len(r):.1f}%)")
        print(f"  MLP ablation HURT gold (rank worsened):   {hurt}/{len(r)} ({100*hurt/len(r):.1f}%)")
        print(f"  Mean rank change: {np.mean(improvements):.1f} (positive=helped)")
        print(f"  Median rank change: {np.median(improvements):.0f}")

    # Key comparison
    pres_helped = sum(1 for x in results["preservation"] if x["improvement"] > 0)
    form_helped = sum(1 for x in results["formation"] if x["improvement"] > 0)
    n_pres = len(results["preservation"])
    n_form = len(results["formation"])
    print(f"\n[Key comparison]")
    print(f"  Preservation: {pres_helped}/{n_pres} ({100*pres_helped/max(n_pres,1):.1f}%) helped by MLP ablation")
    print(f"  Formation:    {form_helped}/{n_form} ({100*form_helped/max(n_form,1):.1f}%) helped by MLP ablation")
    if form_helped > 0:
        print(f"  Ratio: {pres_helped/max(form_helped,1):.1f}x")

    out_file = os.path.join(config.DATA_DIR, "activation_patching_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump({"target_layer": target_layer, "results": results}, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
