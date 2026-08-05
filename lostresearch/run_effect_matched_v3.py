"""Effect-matched v3: match ||Δh|| (hidden state change) instead of Δz."""
import json, os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import config
from run_cross_model import MODELS, get_residual_layers, load_data, is_answer_correct


@torch.no_grad()
def generate_with_ablation(model, tokenizer, prompt_ids, direction, alpha, ablate_layers):
    """h' = h - alpha * proj_dir(h), then generate."""
    layers = get_residual_layers(model)
    device = model.device

    def hook(module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, ()
        last_h = h[0, -1, :]
        denom = (direction @ direction)
        if denom > 1e-10:
            proj = (last_h @ direction) / denom * direction
            h[0, -1, :] = last_h - alpha * proj
        if rest:
            return (h,) + rest
        return h

    hooks = [layers[l].register_forward_hook(hook) for l in ablate_layers]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    for hk in hooks:
        hk.remove()
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def get_proj_norm(model, prompt_ids, direction, ablate_layers):
    """Get ||proj_dir(h)|| at target position for each ablate layer."""
    layers = get_residual_layers(model)
    device = model.device
    norms = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        last_h = h[0, -1, :]
        denom = (direction @ direction)
        if denom > 1e-10:
            proj = (last_h @ direction) / denom * direction
            norms.append(proj.norm().item())

    hooks = [layers[l].register_forward_hook(hook) for l in ablate_layers]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model(input_ids, use_cache=False)
    for hk in hooks:
        hk.remove()
    return norms


@torch.no_grad()
def generate_norm_matched(model, tokenizer, prompt_ids, direction, target_norm, ablate_layers):
    """Ablate direction so per-layer ||Δh|| = target_norm."""
    layers = get_residual_layers(model)
    device = model.device

    def hook(module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, ()
        last_h = h[0, -1, :]
        denom = (direction @ direction)
        if denom > 1e-10:
            proj = (last_h @ direction) / denom * direction
            pn = proj.norm().item()
            if pn > 1e-8:
                alpha = target_norm / pn
            else:
                alpha = 0.0
            h[0, -1, :] = last_h - alpha * proj
        if rest:
            return (h,) + rest
        return h

    hooks = [layers[l].register_forward_hook(hook) for l in ablate_layers]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    for hk in hooks:
        hk.remove()
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def run(model_key, n_samples=20):
    cfg = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"Effect-Matched v3 (||Dh|| matched): {cfg['name']}")
    print(f"{'='*70}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODELS["qwen"]["path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    unembed = model.lm_head.weight
    device = model.device
    ablate_layers = list(range(num_layers - 3, num_layers))

    samples = load_data(tokenizer, model_key, n_per_task=30)
    results = []
    correct_found = 0

    for s in tqdm(samples, desc="Norm-matched"):
        if correct_found >= n_samples:
            break
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        gold_aliases = s["aliases"]

        # Baseline check
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        base_out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        base_gen = tokenizer.decode(base_out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        if not is_answer_correct(base_gen, gold_aliases):
            continue
        correct_found += 1

        gold_dir = unembed[gold_token].detach()

        # Gold ablation (alpha=1) and measure ||proj||
        gold_norms = get_proj_norm(model, prompt_ids, gold_dir, ablate_layers)
        target_norm = np.mean(gold_norms) if gold_norms else 1.0
        gold_gen = generate_with_ablation(model, tokenizer, prompt_ids, gold_dir, 1.0, ablate_layers)
        gold_correct = is_answer_correct(gold_gen, gold_aliases)

        # Second-best (norm-matched)
        final_logits = model(input_ids, use_cache=False).logits[0, -1, :]
        sorted_t = torch.argsort(final_logits, descending=True)
        second_t = sorted_t[0].item() if sorted_t[0].item() != gold_token else sorted_t[1].item()
        second_dir = unembed[second_t].detach()
        second_gen = generate_norm_matched(model, tokenizer, prompt_ids, second_dir, target_norm, ablate_layers)
        second_correct = is_answer_correct(second_gen, gold_aliases)

        # Random token (norm-matched)
        rand_t = np.random.randint(0, unembed.shape[0])
        while rand_t == gold_token:
            rand_t = np.random.randint(0, unembed.shape[0])
        rand_dir = unembed[rand_t].detach()
        rand_gen = generate_norm_matched(model, tokenizer, prompt_ids, rand_dir, target_norm, ablate_layers)
        rand_correct = is_answer_correct(rand_gen, gold_aliases)

        # Gaussian (norm-matched)
        gauss_dir = torch.randn_like(gold_dir)
        gauss_dir = gauss_dir / gauss_dir.norm() * gold_dir.norm()
        gauss_gen = generate_norm_matched(model, tokenizer, prompt_ids, gauss_dir, target_norm, ablate_layers)
        gauss_correct = is_answer_correct(gauss_gen, gold_aliases)

        results.append({
            "id": s["id"], "answer": s["answer"],
            "target_norm": target_norm,
            "gold_flip": not gold_correct,
            "second_flip": not second_correct,
            "random_flip": not rand_correct,
            "gaussian_flip": not gauss_correct,
        })

        if len(results) <= 3:
            print(f"  {s['id']}: norm={target_norm:.1f} gold={not gold_correct} "
                  f"second={not second_correct} random={not rand_correct} gauss={not gauss_correct}")

    # Summary
    n = len(results)
    print(f"\n--- Results (||Dh|| matched, n={n}) ---")
    if n == 0:
        return results
    gf = sum(r["gold_flip"] for r in results)
    sf = sum(r["second_flip"] for r in results)
    rf = sum(r["random_flip"] for r in results)
    gaf = sum(r["gaussian_flip"] for r in results)
    print(f"  Gold direction:    {gf}/{n} ({100*gf/n:.1f}%)")
    print(f"  Second-best:       {sf}/{n} ({100*sf/n:.1f}%)")
    print(f"  Random token:      {rf}/{n} ({100*rf/n:.1f}%)")
    print(f"  Gaussian:          {gaf}/{n} ({100*gaf/n:.1f}%)")
    if gf > sf and gf > rf and gf > gaf:
        print("  => Gold direction flips MORE at same ||Dh|| -> specificity holds")
    elif gf > gaf:
        print("  => Gold > Gaussian but not clearly > second/random")
    else:
        print("  => Gold NOT more effective at same ||Dh||")

    out_file = os.path.join(config.DATA_DIR, f"effect_matched_v3_{model_key}.json")
    with open(out_file, "w") as f:
        json.dump({"model": cfg["name"], "results": results}, f, indent=2)
    print(f"  Saved: {out_file}")
    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    run(args.model, n_samples=args.n)
