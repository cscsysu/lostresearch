"""
Review v4: Matched token-direction control for behavioral necessity

Review 指出: random direction 和任何 token 正交, 0% flips 不意外.
必须用 "随机 token 的 unembedding 方向" 作为更强的对照.

实验:
1. 消融 gold direction → flips (已做: 76%)
2. 消融 random direction → flips (已做: 0%) ← 太弱
3. 消融 random token unembedding direction → flips (新)
4. 消融 second-best token direction → flips (新)
5. 消融 gold-vs-competitor contrast direction (d = w_gold - w_competitor) → flips (新)

如果 gold direction flips 显著高于 random token direction,
说明效果不是 "任何 unembedding 方向都有效", 而是特异于 gold.
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def is_answer_correct(generated, aliases):
    gen_lower = generated.lower().strip()
    for ans in aliases:
        if ans.lower().strip() in gen_lower:
            return True
    return False


@torch.no_grad()
def generate_answer(model, tokenizer, prompt_ids):
    device = model.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids, max_new_tokens=32, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def ablate_and_generate(model, tokenizer, prompt_ids, direction, layers_to_ablate):
    """消融指定方向, 重新生成."""
    layers = get_residual_layers(model)
    device = model.device
    dir_norm = direction.norm()

    def ablate_hook_factory(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            last_h = h[0, -1, :]
            proj = (last_h @ direction) / (direction @ direction) * direction
            h[0, -1, :] = last_h - proj
            if rest:
                return (h,) + rest
            return h
        return hook

    hooks = [layers[l].register_forward_hook(ablate_hook_factory(l)) for l in layers_to_ablate]
    gen_text = generate_answer(model, tokenizer, prompt_ids)
    for h in hooks:
        h.remove()
    return gen_text


def run_matched_controls(model, tokenizer, correct_samples, n_samples=30):
    """5 种消融对照."""
    print("\n" + "=" * 70)
    print("Review v4: Matched Token-Direction Controls")
    print("=" * 70)

    layers = get_residual_layers(model)
    num_layers = len(layers)
    unembed = model.lm_head.weight  # [vocab, hidden]
    device = model.device
    ablate_layers = list(range(num_layers - 3, num_layers))

    results = []
    for i, s in enumerate(tqdm(correct_samples[:n_samples], desc="Matched controls")):
        try:
            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            gold_aliases = s["aliases"]

            # Baseline
            baseline_gen = generate_answer(model, tokenizer, prompt_ids)
            baseline_correct = is_answer_correct(baseline_gen, gold_aliases)
            if not baseline_correct:
                continue

            # 1. Gold direction (已验证)
            gold_dir = unembed[target_token].detach()
            gold_gen = ablate_and_generate(model, tokenizer, prompt_ids, gold_dir, ablate_layers)
            gold_correct = is_answer_correct(gold_gen, gold_aliases)

            # 2. Random direction (同范数, 已验证太弱)
            random_dir = torch.randn_like(gold_dir)
            random_dir = random_dir / random_dir.norm() * gold_dir.norm()
            random_gen = ablate_and_generate(model, tokenizer, prompt_ids, random_dir, ablate_layers)
            random_correct = is_answer_correct(random_gen, gold_aliases)

            # 3. Random token unembedding direction (新, 更强对照)
            # 选一个随机 token, 用它的 unembedding 方向
            vocab_size = unembed.shape[0]
            random_token = np.random.randint(0, vocab_size)
            while random_token == target_token:
                random_token = np.random.randint(0, vocab_size)
            random_token_dir = unembed[random_token].detach()
            random_token_dir = random_token_dir / random_token_dir.norm() * gold_dir.norm()
            random_token_gen = ablate_and_generate(model, tokenizer, prompt_ids, random_token_dir, ablate_layers)
            random_token_correct = is_answer_correct(random_token_gen, gold_aliases)

            # 4. Second-best token direction
            # 找最终层 logits 排第二的 token
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids, use_cache=False)
            final_logits = out.logits[0, -1, :]
            sorted_idx = torch.argsort(final_logits, descending=True)
            second_best = sorted_idx[0].item() if sorted_idx[0].item() != target_token else sorted_idx[1].item()
            second_dir = unembed[second_best].detach()
            second_dir = second_dir / second_dir.norm() * gold_dir.norm()
            second_gen = ablate_and_generate(model, tokenizer, prompt_ids, second_dir, ablate_layers)
            second_correct = is_answer_correct(second_gen, gold_aliases)

            # 5. Gold-vs-competitor contrast direction (d = w_gold - w_competitor)
            # competitor = second best
            contrast_dir = (unembed[target_token] - unembed[second_best]).detach()
            contrast_dir = contrast_dir / contrast_dir.norm() * gold_dir.norm()
            contrast_gen = ablate_and_generate(model, tokenizer, prompt_ids, contrast_dir, ablate_layers)
            contrast_correct = is_answer_correct(contrast_gen, gold_aliases)

            results.append({
                "id": s["id"],
                "answer": s["answer"],
                "baseline_correct": True,
                "gold_correct": gold_correct,
                "random_correct": random_correct,
                "random_token_correct": random_token_correct,
                "second_best_correct": second_correct,
                "contrast_correct": contrast_correct,
                "gold_flipped": not gold_correct,
                "random_flipped": not random_correct,
                "random_token_flipped": not random_token_correct,
                "second_best_flipped": not second_correct,
                "contrast_flipped": not contrast_correct,
            })

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # 分析
    print("\n--- 结果 ---")
    n = len(results)
    print(f"样本数: {n}")
    print()
    print(f"{'Direction':<25} {'Correct':<10} {'Flipped':<10} {'Flip Rate'}")
    print("-" * 55)

    for name, key in [
        ("Gold (target)", "gold_flipped"),
        ("Random (Gaussian)", "random_flipped"),
        ("Random token unembed", "random_token_flipped"),
        ("Second-best token", "second_best_flipped"),
        ("Gold-competitor contrast", "contrast_flipped"),
    ]:
        flipped = sum(1 for r in results if r[key])
        correct = n - flipped
        print(f"{name:<25} {correct:<10} {flipped:<10} {100*flipped/n:.1f}%")

    # 判断
    gold_flips = sum(1 for r in results if r["gold_flipped"])
    random_token_flips = sum(1 for r in results if r["random_token_flipped"])
    second_flips = sum(1 for r in results if r["second_best_flipped"])

    print(f"\n--- 判断 ---")
    if gold_flips > random_token_flips and gold_flips > second_flips:
        print(f"✓ Gold direction 消融导致更多 flips ({gold_flips})")
        print(f"  vs random token ({random_token_flips}) 和 second-best ({second_flips})")
        print(f"  → 效果特异于 gold direction, 不是 '任何 unembedding 方向都有效'")
    else:
        print(f"? Gold direction 未显著优于对照")

    return results


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from data_loader import load_all_datasets, prepare_samples

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE,
    )
    model.eval()

    print("Loading data...")
    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)

    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        return
    with open(results_file) as f:
        all_results = json.load(f)

    # 选正确样本
    correct_samples = []
    for s in prepared:
        result = next((r for r in all_results if r["id"] == s["id"]), None)
        if result and result["final_correct"]:
            correct_samples.append(s)
        if len(correct_samples) >= 30:
            break

    results = run_matched_controls(model, tokenizer, correct_samples, n_samples=30)

    out_file = os.path.join(config.DATA_DIR, "matched_controls_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
