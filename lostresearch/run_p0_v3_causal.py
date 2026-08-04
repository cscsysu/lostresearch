"""
P0-5: 行为层 necessity (correct-to-wrong flips)
P0-6: 多点 patching (分布式证据)

这两个实验需要模型 forward.
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from intervention import ActivationPatcher, get_residual_layers


def run_behavioral_necessity(model, tokenizer, prepared, all_results, n_samples=50):
    """P0-5: 行为层 necessity.

    不能只测 gold logit 下降, 必须测 correct-to-wrong flips.

    实验:
    1. 选正确样本
    2. 消融最后几层的 gold direction
    3. 重新 generate, 看是否变错
    4. 对照: 消融 matched random direction, 看是否不变错
    """
    print("\n" + "=" * 70)
    print("P0-5: Behavioral Necessity (correct-to-wrong flips)")
    print("=" * 70)

    patcher = ActivationPatcher(model, tokenizer)
    layers = get_residual_layers(model)
    unembed = model.lm_head.weight
    device = model.device

    # 选正确样本
    correct_samples = []
    for s in prepared:
        result = next((r for r in all_results if r["id"] == s["id"]), None)
        if result and result["final_correct"]:
            correct_samples.append((s, result))
        if len(correct_samples) >= n_samples:
            break

    print(f"正确样本数: {len(correct_samples)}")

    # 对每个样本, 在最后 3 层消融 gold direction, 然后重新 generate
    results = []
    for i, (s, result) in enumerate(tqdm(correct_samples, desc="Behavioral necessity")):
        try:
            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            gold_aliases = s["aliases"]

            # 1. Baseline: 不消融, 重新 generate (应该答对)
            with torch.no_grad():
                gen_out = model.generate(
                    torch.tensor([prompt_ids], dtype=torch.long, device=device),
                    max_new_tokens=32, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            baseline_gen = tokenizer.decode(gen_out[0][len(prompt_ids):], skip_special_tokens=True).strip()
            baseline_correct = any(a.lower() in baseline_gen.lower() for a in gold_aliases)

            # 2. 消融 gold direction (layer 33-35)
            answer_dir = unembed[target_token].detach()

            def ablate_hook_factory(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        rest = output[1:]
                    else:
                        h = output
                        rest = ()
                    # 消融最后 token 位置的 gold direction
                    last_h = h[0, -1, :]
                    proj = (last_h @ answer_dir) / (answer_dir @ answer_dir) * answer_dir
                    h[0, -1, :] = last_h - proj
                    if rest:
                        return (h,) + rest
                    return h
                return hook

            hooks = []
            for l in [33, 34, 35]:
                hooks.append(layers[l].register_forward_hook(ablate_hook_factory(l)))
            with torch.no_grad():
                gen_out = model.generate(
                    torch.tensor([prompt_ids], dtype=torch.long, device=device),
                    max_new_tokens=32, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for h in hooks:
                h.remove()
            ablated_gen = tokenizer.decode(gen_out[0][len(prompt_ids):], skip_special_tokens=True).strip()
            ablated_correct = any(a.lower() in ablated_gen.lower() for a in gold_aliases)

            # 3. 对照: 消融 random direction (同范数)
            random_dir = torch.randn_like(answer_dir)
            random_dir = random_dir / random_dir.norm() * answer_dir.norm()

            def random_hook_factory(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        rest = output[1:]
                    else:
                        h = output
                        rest = ()
                    last_h = h[0, -1, :]
                    proj = (last_h @ random_dir) / (random_dir @ random_dir) * random_dir
                    h[0, -1, :] = last_h - proj
                    if rest:
                        return (h,) + rest
                    return h
                return hook

            hooks = []
            for l in [33, 34, 35]:
                hooks.append(layers[l].register_forward_hook(random_hook_factory(l)))
            with torch.no_grad():
                gen_out = model.generate(
                    torch.tensor([prompt_ids], dtype=torch.long, device=device),
                    max_new_tokens=32, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for h in hooks:
                h.remove()
            random_gen = tokenizer.decode(gen_out[0][len(prompt_ids):], skip_special_tokens=True).strip()
            random_correct = any(a.lower() in random_gen.lower() for a in gold_aliases)

            results.append({
                "id": s["id"],
                "answer": s["answer"],
                "baseline_gen": baseline_gen[:30],
                "baseline_correct": baseline_correct,
                "ablated_gen": ablated_gen[:30],
                "ablated_correct": ablated_correct,
                "random_gen": random_gen[:30],
                "random_correct": random_correct,
                "flipped_to_wrong": baseline_correct and not ablated_correct,
                "random_flipped": baseline_correct and not random_correct,
            })

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # 分析
    print("\n--- 结果 ---")
    n = len(results)
    baseline_correct = sum(1 for r in results if r["baseline_correct"])
    ablated_correct = sum(1 for r in results if r["ablated_correct"])
    random_correct = sum(1 for r in results if r["random_correct"])
    flipped = sum(1 for r in results if r["flipped_to_wrong"])
    random_flipped = sum(1 for r in results if r["random_flipped"])

    print(f"Baseline correct: {baseline_correct}/{n}")
    print(f"Ablated correct:   {ablated_correct}/{n} ({baseline_correct - ablated_correct} flipped to wrong)")
    print(f"Random ablation:  {random_correct}/{n} ({baseline_correct - random_correct} flipped)")
    print(f"\nGold direction flips: {flipped}/{baseline_correct}")
    print(f"Random direction flips: {random_flipped}/{baseline_correct}")

    if flipped > random_flipped:
        print(f"\n✓ Gold direction 消融导致更多 correct-to-wrong flips ({flipped} > {random_flipped})")
        print("  → 行为层 necessity 成立")
    else:
        print(f"\n? Gold direction 未导致显著更多 flips")

    return results


def run_multi_point_patching(model, tokenizer, prepared, all_results, n_samples=30):
    """P0-6: 多点 patching.

    如果多点 patching 显著优于单点, "分布式"结论成立.
    """
    print("\n" + "=" * 70)
    print("P0-6: Multi-Point Patching (Distributed Evidence)")
    print("=" * 70)

    patcher = ActivationPatcher(model, tokenizer)
    device = model.device

    results = []
    for i, s in enumerate(tqdm(prepared[:n_samples], desc="Multi-point patching")):
        try:
            result = next((r for r in all_results if r["id"] == s["id"]), None)
            if result is None:
                continue

            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]

            # Clean run
            clean_hiddens = patcher.collect_hiddens(prompt_ids)

            # Corrupted run (hint wrong answer)
            wrong_answer = "unknown"
            corr_text = s["question"] + "\nHint: the answer is " + wrong_answer + ". Answer with just the answer."
            from data_loader import build_prompt
            corr_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": corr_text}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=config.ENABLE_THINKING,
            )
            corr_ids = tokenizer.encode(corr_prompt, add_special_tokens=False)
            corr_ids = corr_ids[-config.MAX_PROMPT_LEN:]

            # Corrupted baseline
            with torch.no_grad():
                corr_out = model(torch.tensor([corr_ids], dtype=torch.long, device=device), use_cache=False)
            corr_target_lp = F.log_softmax(corr_out.logits[0, -1, :], dim=-1)[target_token].item()

            # 1. 单点 patch (layer 33)
            single = patcher.patched_forward(
                corr_ids, clean_hiddens, patch_layer=33,
                patch_pos=-1, clean_prompt_ids=prompt_ids,
                target_token_id=target_token,
            )
            single_recovery = single["target_logprob"] - corr_target_lp

            # 2. 多点 patch (layer 30-35, 6 层)
            # 用 patched_forward_with_logit_lens 的方式, 逐层 patch
            # 简化: 在 hook 里对多层都做替换
            multi_recovery = 0
            try:
                # 找公共前缀长度
                common_len = patcher._find_common_prefix_len(prompt_ids, corr_ids)
                patch_len = min(common_len, len(prompt_ids))

                # 注册多层 hook
                patch_layers = [30, 31, 32, 33, 34, 35]
                hooks = []
                for l in patch_layers:
                    def make_hook(layer_idx):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0]
                                rest = output[1:]
                            else:
                                h = output
                                rest = ()
                            if h.dim() == 3:
                                for pos in range(min(patch_len, h.shape[1])):
                                    h[0, pos] = clean_hiddens[layer_idx][pos].to(h.device).to(h.dtype)
                            if rest:
                                return (h,) + rest
                            return h
                        return hook
                    hooks.append(get_residual_layers(model)[layer_idx].register_forward_hook(make_hook(l)))

                with torch.no_grad():
                    multi_out = model(torch.tensor([corr_ids], dtype=torch.long, device=device), use_cache=False)
                for h in hooks:
                    h.remove()
                multi_target_lp = F.log_softmax(multi_out.logits[0, -1, :], dim=-1)[target_token].item()
                multi_recovery = multi_target_lp - corr_target_lp
            except Exception as e:
                print(f"  Multi-point error on {s['id']}: {e}")

            # 3. Norm-matched random (单点, 对照)
            random_dir = torch.randn_like(clean_hiddens[33][-1])
            random_dir = random_dir / random_dir.norm() * clean_hiddens[33][-1].norm()
            random_hiddens = {33: clean_hiddens[33].clone()}
            random_hiddens[33][-1] = random_dir
            random_patched = patcher.patched_forward(
                corr_ids, random_hiddens, patch_layer=33,
                patch_pos=-1, target_token_id=target_token,
            )
            random_recovery = random_patched["target_logprob"] - corr_target_lp

            results.append({
                "id": s["id"],
                "corrupted_baseline": corr_target_lp,
                "single_recovery": single_recovery,
                "multi_recovery": multi_recovery,
                "random_recovery": random_recovery,
            })

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # 分析
    print("\n--- 结果 ---")
    singles = [r["single_recovery"] for r in results]
    multis = [r["multi_recovery"] for r in results]
    randoms = [r["random_recovery"] for r in results]

    print(f"Single-point recovery: mean={np.mean(singles):.3f}")
    print(f"Multi-point recovery:  mean={np.mean(multis):.3f}")
    print(f"Random recovery:       mean={np.mean(randoms):.3f}")

    if np.mean(multis) > np.mean(singles) + 0.5 and np.mean(multis) > np.mean(randoms) + 0.5:
        print(f"\n✓ 多点 patching 显著优于单点和随机")
        print("  → 信号是分布式的, 单点不足以恢复")
    elif np.mean(multis) > np.mean(singles):
        print(f"\n? 多点略优于单点, 但不显著")
    else:
        print(f"\n✗ 多点 patching 未优于单点")

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

    behavioral = run_behavioral_necessity(model, tokenizer, prepared, all_results, n_samples=50)
    multi_point = run_multi_point_patching(model, tokenizer, prepared, all_results, n_samples=30)

    out = {"behavioral_necessity": behavioral, "multi_point_patching": multi_point}
    out_file = os.path.join(config.DATA_DIR, "p0_v3_causal_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
