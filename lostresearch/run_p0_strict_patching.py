"""
P0 实验 5+6: 严格 patching 对照 + Mediation analysis

5 个对照:
1. Random-example patching: 从无关样本注入同层激活
2. Wrong-layer patching: 在错误层注入
3. Wrong-position patching: 在错误位置注入
4. Norm-matched random direction: 注入相同范数的随机方向
5. CIS-direction-only patching: 只保留与答案竞争方向相关的分量

Mediation:
patch → ΔCIS_{l+1:L} → ΔP(correct) → Δaccuracy
"""
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from intervention import ActivationPatcher, get_residual_layers


def get_final_norm_and_unembed(model):
    return model.model.norm, model.lm_head.weight


def run_strict_patching_controls(model, tokenizer, prepared, all_results, n_samples=30):
    """5 个严格对照 + 主实验."""
    print("\n" + "=" * 70)
    print("P0-5: Strict Patching Controls")
    print("=" * 70)

    patcher = ActivationPatcher(model, tokenizer)
    layers = get_residual_layers(model)
    num_layers = len(layers)
    final_norm, unembed = get_final_norm_and_unembed(model)
    device = model.device

    key_layers = [28, 31, 33, 35]  # 关键层

    results = []
    for i, s in enumerate(tqdm(prepared[:n_samples], desc="Strict patching")):
        try:
            result = next((r for r in all_results if r["id"] == s["id"]), None)
            if result is None:
                continue

            cis = result["cis"]
            if len(cis) < 4:
                continue

            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            gen_token = result.get("generated_token_ids", [target_token])[0]

            # 1. Clean run: 采集 hiddens
            clean_hiddens = patcher.collect_hiddens(prompt_ids)

            # 2. Corrupted run: 用 hint wrong answer
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

            # 3. Corrupted baseline (不 patch)
            with torch.no_grad():
                corr_out = model(torch.tensor([corr_ids], dtype=torch.long, device=device), use_cache=False)
            corr_logits = corr_out.logits[0, -1, :]
            corr_target_lp = F.log_softmax(corr_logits, dim=-1)[target_token].item()

            # 4. 5 个对照 + 主实验, 在每个 key layer
            layer_results = {}
            for layer in key_layers:
                experiments = {}

                # A. 主实验: clean → corrupted (同一样本)
                patched = patcher.patched_forward(
                    corr_ids, clean_hiddens, patch_layer=layer,
                    patch_pos=-1, clean_prompt_ids=prompt_ids,
                    target_token_id=target_token,
                )
                experiments["clean_to_corr"] = patched["target_logprob"]

                # B. Random-example: 用另一个样本的 clean hiddens
                other_idx = (i + 1) % len(prepared)
                other = prepared[other_idx]
                other_hiddens = patcher.collect_hiddens(other["prompt_ids"])
                patched_random = patcher.patched_forward(
                    corr_ids, other_hiddens, patch_layer=layer,
                    patch_pos=-1, clean_prompt_ids=other["prompt_ids"],
                    target_token_id=target_token,
                )
                experiments["random_example"] = patched_random["target_logprob"]

                # C. Wrong-layer: 在 layer 0 注入 (应该无效)
                patched_wrong = patcher.patched_forward(
                    corr_ids, clean_hiddens, patch_layer=0,
                    patch_pos=-1, clean_prompt_ids=prompt_ids,
                    target_token_id=target_token,
                )
                experiments["wrong_layer"] = patched_wrong["target_logprob"]

                # D. Norm-matched random direction
                clean_h = clean_hiddens[layer][-1]
                random_dir = torch.randn_like(clean_h)
                random_dir = random_dir / random_dir.norm() * clean_h.norm()
                random_hiddens = {layer: clean_hiddens[layer].clone()}
                random_hiddens[layer][-1] = random_dir
                patched_norm = patcher.patched_forward(
                    corr_ids, random_hiddens, patch_layer=layer,
                    patch_pos=-1, target_token_id=target_token,
                )
                experiments["norm_matched_random"] = patched_norm["target_logprob"]

                # E. CIS-direction-only: 只保留 answer direction 分量
                answer_dir = unembed[target_token].detach()
                clean_h_last = clean_hiddens[layer][-1]
                proj = (clean_h_last @ answer_dir) / (answer_dir @ answer_dir) * answer_dir
                cis_hiddens = {layer: clean_hiddens[layer].clone()}
                cis_hiddens[layer][-1] = proj
                patched_cis = patcher.patched_forward(
                    corr_ids, cis_hiddens, patch_layer=layer,
                    patch_pos=-1, target_token_id=target_token,
                )
                experiments["cis_direction_only"] = patched_cis["target_logprob"]

                layer_results[layer] = {
                    "corrupted_baseline": corr_target_lp,
                    **experiments,
                    "recovery_clean": experiments["clean_to_corr"] - corr_target_lp,
                    "recovery_random": experiments["random_example"] - corr_target_lp,
                    "recovery_wrong_layer": experiments["wrong_layer"] - corr_target_lp,
                    "recovery_norm_matched": experiments["norm_matched_random"] - corr_target_lp,
                    "recovery_cis_only": experiments["cis_direction_only"] - corr_target_lp,
                }

            results.append({
                "id": s["id"],
                "question": s["question"][:50],
                "answer": s["answer"],
                "final_correct": result["final_correct"],
                "layer_results": layer_results,
            })

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # 分析
    print("\n" + "=" * 70)
    print("STRICT PATCHING ANALYSIS")
    print("=" * 70)

    for layer in key_layers:
        print(f"\n  Layer {layer}:")
        clean_recs = [r["layer_results"][layer]["recovery_clean"] for r in results if layer in r["layer_results"]]
        random_recs = [r["layer_results"][layer]["recovery_random"] for r in results if layer in r["layer_results"]]
        wrong_recs = [r["layer_results"][layer]["recovery_wrong_layer"] for r in results if layer in r["layer_results"]]
        norm_recs = [r["layer_results"][layer]["recovery_norm_matched"] for r in results if layer in r["layer_results"]]
        cis_recs = [r["layer_results"][layer]["recovery_cis_only"] for r in results if layer in r["layer_results"]]

        print(f"    clean→corr:    mean={np.mean(clean_recs):.3f}")
        print(f"    random example: mean={np.mean(random_recs):.3f}")
        print(f"    wrong layer:    mean={np.mean(wrong_recs):.3f}")
        print(f"    norm-matched:   mean={np.mean(norm_recs):.3f}")
        print(f"    CIS-only:       mean={np.mean(cis_recs):.3f}")

        # 判据: clean 应该 > random/wrong/norm
        if np.mean(clean_recs) > np.mean(random_recs) and np.mean(clean_recs) > np.mean(norm_recs):
            print(f"    ✓ Clean patching 显著优于对照")
        else:
            print(f"    ? Clean patching 未优于对照")

    return results


def run_mediation_analysis(model, tokenizer, prepared, all_results, n_samples=30):
    """Mediation: patch → ΔCIS_{l+1:L} → ΔP(correct) → Δaccuracy."""
    print("\n" + "=" * 70)
    print("P0-6: Mediation Analysis")
    print("=" * 70)

    patcher = ActivationPatcher(model, tokenizer)
    layers = get_residual_layers(model)
    num_layers = len(layers)
    final_norm, unembed = get_final_norm_and_unembed(model)
    device = model.device

    results = []
    for i, s in enumerate(tqdm(prepared[:n_samples], desc="Mediation")):
        try:
            result = next((r for r in all_results if r["id"] == s["id"]), None)
            if result is None:
                continue

            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]

            # Clean run
            clean_hiddens = patcher.collect_hiddens(prompt_ids)

            # Corrupted run
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

            # Patched forward (layer 33), 追踪后续层 CIS
            patch_layer = 33

            # 1. Corrupted baseline CIS per layer
            with torch.no_grad():
                corr_out = model(torch.tensor([corr_ids], dtype=torch.long, device=device), use_cache=False)
            # 采集 corrupted 的每层 hidden
            corr_hiddens = {}
            def make_hook_corrupted(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    corr_hiddens[idx] = h[0, -1, :].detach().clone()
                return hook
            hooks = [layer.register_forward_hook(make_hook_corrupted(l)) for l, layer in enumerate(layers)]
            with torch.no_grad():
                model(torch.tensor([corr_ids], dtype=torch.long, device=device), use_cache=False)
            for h in hooks:
                h.remove()

            # 2. Patched run (patch layer 33), 追踪后续层 CIS
            patched_hiddens = {}
            def make_hook_patched(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    patched_hiddens[idx] = h[0, -1, :].detach().clone()
                return hook

            # patch layer `patch_layer`
            clean_h = clean_hiddens[patch_layer][-1]
            def patch_hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    rest = output[1:]
                else:
                    h = output
                    rest = ()
                if h.dim() == 3:
                    h[0, -1] = clean_h.to(h.device).to(h.dtype)
                elif h.dim() == 2:
                    h[0] = clean_h.to(h.device).to(h.dtype)
                if rest:
                    return (h,) + rest
                return h

            patch_handle = layers[patch_layer].register_forward_hook(patch_hook)
            hooks2 = [layer.register_forward_hook(make_hook_patched(l)) for l, layer in enumerate(layers)]
            with torch.no_grad():
                patched_out = model(torch.tensor([corr_ids], dtype=torch.long, device=device), use_cache=False)
            for h in hooks2:
                h.remove()
            patch_handle.remove()

            # 3. 计算每层 CIS (corrupted vs patched)
            cis_corrupted = []
            cis_patched = []
            for l in range(num_layers):
                h_corr = corr_hiddens[l].to(device)
                h_patch = patched_hiddens[l].to(device)

                lp_corr = F.log_softmax(F.linear(final_norm(h_corr), unembed), dim=-1)
                lp_patch = F.log_softmax(F.linear(final_norm(h_patch), unembed), dim=-1)

                cis_corrupted.append((lp_corr[target_token] - lp_corr[target_token]).item())  # 0
                cis_patched.append((lp_patch[target_token] - lp_patch[target_token]).item())

            # 简化: 只看 patch 后续层 (l > patch_layer) 的 target logprob
            target_lp_corrupted = []
            target_lp_patched = []
            for l in range(num_layers):
                h_corr = corr_hiddens[l].to(device)
                h_patch = patched_hiddens[l].to(device)
                lp_corr = F.log_softmax(F.linear(final_norm(h_corr), unembed), dim=-1)
                lp_patch = F.log_softmax(F.linear(final_norm(h_patch), unembed), dim=-1)
                target_lp_corrupted.append(lp_corr[target_token].item())
                target_lp_patched.append(lp_patch[target_token].item())

            # 最终层 logprob
            final_lp_corr = target_lp_corrupted[-1]
            final_lp_patch = target_lp_patched[-1]
            delta_lp = final_lp_patch - final_lp_corr

            # 后续层 CIS 变化 (l > patch_layer)
            post_patch_delta = [target_lp_patched[l] - target_lp_corrupted[l]
                                for l in range(patch_layer+1, num_layers)]
            mean_post_delta = np.mean(post_patch_delta) if post_patch_delta else 0

            results.append({
                "id": s["id"],
                "patch_layer": patch_layer,
                "final_lp_corrupted": final_lp_corr,
                "final_lp_patched": final_lp_patch,
                "delta_final_lp": delta_lp,
                "mean_post_patch_delta": mean_post_delta,
                "target_lp_corrupted_per_layer": target_lp_corrupted,
                "target_lp_patched_per_layer": target_lp_patched,
            })

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # 分析
    print("\n  Mediation Summary:")
    delta_finals = [r["delta_final_lp"] for r in results]
    delta_posts = [r["mean_post_patch_delta"] for r in results]
    print(f"  Mean Δ(final target logprob): {np.mean(delta_finals):.3f}")
    print(f"  Mean Δ(post-patch layers): {np.mean(delta_posts):.3f}")

    if np.mean(delta_posts) > 0.5 and np.mean(delta_finals) > 0:
        print("  ✓ Patch 提高了后续层 target logprob → mediation 链成立")
    elif np.mean(delta_posts) > 0.5:
        print("  ? Patch 提高了中间层但未传到最终层")
    else:
        print("  ✗ Patch 未提高后续层 → 无 mediation")

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

    strict_results = run_strict_patching_controls(model, tokenizer, prepared, all_results, n_samples=30)
    mediation_results = run_mediation_analysis(model, tokenizer, prepared, all_results, n_samples=30)

    out = {"strict_patching": strict_results, "mediation": mediation_results}
    out_file = os.path.join(config.DATA_DIR, "p0_strict_patching_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
