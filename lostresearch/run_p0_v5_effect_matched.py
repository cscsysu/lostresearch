"""
Review v5: Effect-matched control for behavioral necessity

核心问题: 消融 gold direction 必然降低 gold logit (因为 z = w^T h).
random token 不降 gold logit 所以 0% flips. 这不公平.

解决: 找一个"非 gold 的方向", 但消融后 gold logit 下降幅度相同,
然后比较 flips.

方法:
1. 消融 gold direction, 记录 gold logit 下降量 Δz
2. 在 unembedding 空间搜索一个方向 d, 使消融后 gold logit 下降相同 Δz
   但 d ≠ w_gold
3. 用 d 做消融, 看 flips

近似实现:
- 选若干个非 gold token 的 unembedding 方向
- 对每个方向, 先消融, 测 gold logit 下降量
- 选下降量最接近 gold 的那个作为 effect-matched control
- 比较两组的 flips
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
def get_gold_logit_after_ablation(model, tokenizer, prompt_ids, direction, ablate_layers):
    """消融指定方向, 返回 gold logit (不 generate, 只 forward)."""
    layers = get_residual_layers(model)
    device = model.device

    def ablate_hook_factory(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            last_h = h[0, -1, :]
            norm_d = direction.norm()
            if norm_d > 0:
                proj = (last_h @ direction) / (direction @ direction) * direction
                h[0, -1, :] = last_h - proj
            if rest:
                return (h,) + rest
            return h
        return hook

    hooks = [layers[l].register_forward_hook(ablate_hook_factory(l)) for l in ablate_layers]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    return out.logits[0, -1, :]


@torch.no_grad()
def generate_after_ablation(model, tokenizer, prompt_ids, direction, ablate_layers):
    """消融指定方向, 生成答案."""
    layers = get_residual_layers(model)
    device = model.device

    def ablate_hook_factory(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            last_h = h[0, -1, :]
            if direction.norm() > 0:
                proj = (last_h @ direction) / (direction @ direction) * direction
                h[0, -1, :] = last_h - proj
            if rest:
                return (h,) + rest
            return h
        return hook

    hooks = [layers[l].register_forward_hook(ablate_hook_factory(l)) for l in ablate_layers]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids, max_new_tokens=32, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    for h in hooks:
        h.remove()
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_effect_matched_control(model, tokenizer, correct_samples, n_samples=20):
    """effect-matched control 实验."""
    print("\n" + "=" * 70)
    print("Review v5: Effect-Matched Control")
    print("=" * 70)

    layers = get_residual_layers(model)
    unembed = model.lm_head.weight
    device = model.device
    num_layers = len(layers)
    ablate_layers = list(range(num_layers - 3, num_layers))

    results = []
    for i, s in enumerate(tqdm(correct_samples[:n_samples], desc="Effect-matched")):
        try:
            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            gold_aliases = s["aliases"]

            # 1. Baseline gold logit (不消融)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                baseline_out = model(input_ids, use_cache=False)
            baseline_gold_logit = baseline_out.logits[0, -1, target_token].item()

            # 验证 baseline 正确
            baseline_gen = tokenizer.decode(
                model.generate(input_ids, max_new_tokens=32, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)[0][input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            if not is_answer_correct(baseline_gen, gold_aliases):
                continue

            # 2. Gold direction 消融: 测 gold logit 下降量
            gold_dir = unembed[target_token].detach()
            gold_after = get_gold_logit_after_ablation(model, tokenizer, prompt_ids, gold_dir, ablate_layers)
            gold_delta_z = baseline_gold_logit - gold_after[target_token].item()

            # 3. 搜索 effect-matched direction
            # 选若干个高 logit token (但不是 gold)
            final_logits = baseline_out.logits[0, -1, :]
            top_tokens = torch.argsort(final_logits, descending=True)[:20]

            best_match_dir = None
            best_match_delta = float('inf')
            best_match_token = None

            for tid in top_tokens:
                tid = tid.item()
                if tid == target_token:
                    continue
                test_dir = unembed[tid].detach()
                test_after = get_gold_logit_after_ablation(model, tokenizer, prompt_ids, test_dir, ablate_layers)
                test_delta = baseline_gold_logit - test_after[target_token].item()
                if abs(test_delta - gold_delta_z) < abs(best_match_delta - gold_delta_z):
                    best_match_delta = test_delta
                    best_match_dir = test_dir
                    best_match_token = tid

            if best_match_dir is None:
                continue

            # 4. Gold direction generate
            gold_gen = generate_after_ablation(model, tokenizer, prompt_ids, gold_dir, ablate_layers)
            gold_correct = is_answer_correct(gold_gen, gold_aliases)

            # 5. Effect-matched direction generate
            matched_gen = generate_after_ablation(model, tokenizer, prompt_ids, best_match_dir, ablate_layers)
            matched_correct = is_answer_correct(matched_gen, gold_aliases)

            results.append({
                "id": s["id"],
                "answer": s["answer"],
                "baseline_gold_logit": baseline_gold_logit,
                "gold_delta_z": gold_delta_z,
                "matched_delta_z": best_match_delta,
                "matched_token_id": best_match_token,
                "gold_correct": gold_correct,
                "matched_correct": matched_correct,
                "gold_flipped": not gold_correct,
                "matched_flipped": not matched_correct,
                "delta_z_difference": abs(gold_delta_z - best_match_delta),
            })

            if i < 3:
                print(f"  [diag] {s['id']}: gold Δz={gold_delta_z:.2f}, "
                      f"matched Δz={best_match_delta:.2f} (token {best_match_token}), "
                      f"diff={abs(gold_delta_z - best_match_delta):.2f}")

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # 分析
    print("\n--- 结果 ---")
    n = len(results)
    print(f"样本数: {n}")
    print(f"Gold Δz: mean={np.mean([r['gold_delta_z'] for r in results]):.3f}")
    print(f"Matched Δz: mean={np.mean([r['matched_delta_z'] for r in results]):.3f}")
    print(f"Δz 差异: mean={np.mean([r['delta_z_difference'] for r in results]):.3f}")

    gold_flips = sum(1 for r in results if r["gold_flipped"])
    matched_flips = sum(1 for r in results if r["matched_flipped"])

    print(f"\nGold direction flips: {gold_flips}/{n} ({100*gold_flips/n:.1f}%)")
    print(f"Effect-matched flips: {matched_flips}/{n} ({100*matched_flips/n:.1f}%)")

    if gold_flips > matched_flips:
        print(f"\n✓ Gold direction 在相同 Δz 下导致更多 flips ({gold_flips} > {matched_flips})")
        print(f"  → 效果特异于 gold direction, 不是 '任何方向降 gold logit 都能 flip'")
    elif gold_flips == matched_flips:
        print(f"\n? 两组相同 → 可能是 '任何降 gold logit 的干预都能 flip'")
    else:
        print(f"\n? Matched 对照反而更多 flips → 需要分析")

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

    correct_samples = []
    for s in prepared:
        result = next((r for r in all_results if r["id"] == s["id"]), None)
        if result and result["final_correct"]:
            correct_samples.append(s)
        if len(correct_samples) >= 20:
            break

    results = run_effect_matched_control(model, tokenizer, correct_samples, n_samples=20)

    out_file = os.path.join(config.DATA_DIR, "effect_matched_control_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
