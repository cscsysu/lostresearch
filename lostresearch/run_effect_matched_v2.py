"""
Effect-matched control (改进版): 真正匹配 gold-logit 下降量.

审稿人批评: 之前 matched Δz=11.96 只有 gold Δz=27.47 的一半, 不能叫 effect-matched.

本脚本修正:
1. 对非 gold direction 施加放大系数 scale, 使消融后 gold-logit 下降量
   与 gold direction 下降量相同 (Δz_match ≈ Δz_gold).
2. 在匹配后比较 flip rate.
3. 支持多个模型 (--model).

用法: python run_effect_matched_v2.py --model qwen
      python run_effect_matched_v2.py --model qwen4b
      python run_effect_matched_v2.py --model qwen14b
"""
import json
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from run_cross_model import MODELS, get_residual_layers, load_data, is_answer_correct


@torch.no_grad()
def forward_logits(model, prompt_ids):
    device = model.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(input_ids, use_cache=False)
    return out.logits[0, -1, :]


@torch.no_grad()
def ablate_forward_logits(model, prompt_ids, direction, alpha, ablate_layers):
    """按 alpha 比例移除 direction 方向的投影, 返回 logits.

    h' = h - alpha * proj_dir(h)
    alpha=1 完全移除该方向; alpha>1 过移除.
    """
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
            denom = (direction @ direction)
            if denom > 0:
                proj = (last_h @ direction) / denom * direction
                h[0, -1, :] = last_h - alpha * proj
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
def ablate_generate(model, tokenizer, prompt_ids, direction, alpha, ablate_layers):
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
            denom = (direction @ direction)
            if denom > 0:
                proj = (last_h @ direction) / denom * direction
                h[0, -1, :] = last_h - alpha * proj
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


def find_match_alpha(model, tokenizer, prompt_ids, gold_token,
                     test_dir, gold_delta_z, ablate_layers,
                     n_iter=15, lo=0.1, hi=50.0):
    """二分搜索 alpha, 使移除 test_dir 的 alpha 比例投影后 gold-logit 下降 ≈ gold_delta_z."""
    baseline_gold = forward_logits(model, prompt_ids)[gold_token].item()

    def delta_at(alpha):
        logits = ablate_forward_logits(model, prompt_ids, test_dir, alpha, ablate_layers)
        return baseline_gold - logits[gold_token].item()

    # 检查范围
    d_hi = delta_at(hi)
    if d_hi < gold_delta_z:
        return hi, d_hi, abs(d_hi - gold_delta_z)
    d_lo = delta_at(lo)
    if d_lo > gold_delta_z:
        return lo, d_lo, abs(d_lo - gold_delta_z)

    # 二分
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        d_mid = delta_at(mid)
        if d_mid < gold_delta_z:
            lo = mid
        else:
            hi = mid
    final_alpha = (lo + hi) / 2
    final_delta = delta_at(final_alpha)
    return final_alpha, final_delta, abs(final_delta - gold_delta_z)


def run_effect_matched(model_key, n_samples=20):
    cfg = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"Effect-Matched Control (v2): {cfg['name']}")
    print(f"{'='*70}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    except Exception as e:
        print(f"  tokenizer load failed ({e}), trying 8B tokenizer fallback...")
        tokenizer = AutoTokenizer.from_pretrained(MODELS["qwen"]["path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    unembed = model.lm_head.weight
    device = model.device
    ablate_layers = list(range(num_layers - 3, num_layers))

    # 加载数据, 找正确样本
    samples = load_data(tokenizer, model_key, n_per_task=20)
    results = []
    correct_found = 0

    for s in tqdm(samples, desc="Effect-matched"):
        if correct_found >= n_samples:
            break
        prompt_ids = s["prompt_ids"]
        gold_token = s["primary_answer_ids"][0]
        gold_aliases = s["aliases"]

        # Baseline: 零方向 + alpha 0, 相当于不消融
        baseline_gen = ablate_generate(model, tokenizer, prompt_ids,
                                       torch.zeros(unembed.shape[1], device=device),
                                       0.0, ablate_layers)
        if not is_answer_correct(baseline_gen, gold_aliases):
            continue

        correct_found += 1

        # 1. Gold direction 消融
        gold_dir = unembed[gold_token].detach()
        gold_after = ablate_forward_logits(model, prompt_ids, gold_dir, 1.0, ablate_layers)
        gold_delta_z = forward_logits(model, prompt_ids)[gold_token].item() - gold_after[gold_token].item()

        # 2. 选对照方向 (根据 control 参数)
        control_dirs = {}
        final_logits = forward_logits(model, prompt_ids)
        sorted_tokens = torch.argsort(final_logits, descending=True)

        # second-best: 最终层 logit 第二高的 token
        second_token = None
        for t in sorted_tokens:
            if t.item() != gold_token:
                second_token = t.item()
                break
        control_dirs["second_best"] = (second_token, unembed[second_token].detach())

        # random token: 随机选一个非 gold token
        vocab_size = unembed.shape[0]
        random_token = np.random.randint(0, vocab_size)
        while random_token == gold_token:
            random_token = np.random.randint(0, vocab_size)
        control_dirs["random"] = (random_token, unembed[random_token].detach())

        # 3. 对每个对照方向做 alpha 匹配 + 生成
        gold_gen = ablate_generate(model, tokenizer, prompt_ids, gold_dir, 1.0, ablate_layers)
        gold_correct = is_answer_correct(gold_gen, gold_aliases)

        entry = {
            "id": s["id"],
            "answer": s["answer"],
            "gold_delta_z": gold_delta_z,
            "gold_correct": gold_correct,
            "gold_flipped": not gold_correct,
            "controls": {},
        }

        for ctrl_name, (ctrl_token, ctrl_dir) in control_dirs.items():
            match_alpha, match_delta, match_err = find_match_alpha(
                model, tokenizer, prompt_ids, gold_token,
                ctrl_dir, gold_delta_z, ablate_layers)
            matched_gen = ablate_generate(model, tokenizer, prompt_ids, ctrl_dir, match_alpha, ablate_layers)
            matched_correct = is_answer_correct(matched_gen, gold_aliases)
            entry["controls"][ctrl_name] = {
                "token": ctrl_token,
                "match_delta_z": match_delta,
                "match_err": match_err,
                "match_alpha": match_alpha,
                "correct": matched_correct,
                "flipped": not matched_correct,
            }

        results.append(entry)

        if len(results) <= 3:
            print(f"  [diag] {s['id']}: gold Δz={gold_delta_z:.2f}, "
                  f"gold_flip={not gold_correct}")
            for ctrl_name, cd in entry["controls"].items():
                print(f"    {ctrl_name}: Δz={cd['match_delta_z']:.2f} "
                      f"(err={cd['match_err']:.2f}), alpha={cd['match_alpha']:.2f}, "
                      f"flip={cd['flipped']}")

    # 分析
    print("\n--- 结果 ---")
    n = len(results)
    print(f"样本数: {n}")
    if n == 0:
        print("  ! 没有正确样本")
        return results

    gold_flips = sum(1 for r in results if r["gold_flipped"])
    print(f"Gold direction flips: {gold_flips}/{n} ({100*gold_flips/n:.1f}%)")

    # 每个对照方向单独统计
    for ctrl_name in ["second_best", "random"]:
        ctrl_flips = sum(1 for r in results
                         if r["controls"].get(ctrl_name, {}).get("flipped", False))
        ctrl_err = np.mean([r["controls"][ctrl_name]["match_err"]
                            for r in results if ctrl_name in r["controls"]])
        ctrl_alpha = np.mean([r["controls"][ctrl_name]["match_alpha"]
                              for r in results if ctrl_name in r["controls"]])
        print(f"\n[{ctrl_name}] control flips: {ctrl_flips}/{n} ({100*ctrl_flips/n:.1f}%)")
        print(f"  Δz match error: mean={ctrl_err:.2f}")
        print(f"  alpha: mean={ctrl_alpha:.2f}")
        if ctrl_err < 2.0:
            print(f"  ✓ Δz 匹配充分 (error < 2.0)")
        else:
            print(f"  ? Δz 匹配不充分 (error={ctrl_err:.2f})")
        if gold_flips > ctrl_flips:
            print(f"  ✓ 相同 Δz 下 gold 翻转更多 ({gold_flips} > {ctrl_flips})")
        elif gold_flips == ctrl_flips:
            print(f"  ? 两者相同")
        else:
            print(f"  ? matched 反而更多 ({ctrl_flips} > {gold_flips})")

    # 保存
    out_file = os.path.join(config.DATA_DIR, f"effect_matched_v2_{model_key}.json")
    with open(out_file, "w") as f:
        json.dump({"model": cfg["name"], "results": results}, f, indent=2)
    print(f"Saved: {out_file}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()))
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    run_effect_matched(args.model, n_samples=args.n)
