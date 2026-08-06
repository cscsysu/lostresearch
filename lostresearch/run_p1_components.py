"""
P1: Component-level Decomposition

分解信号衰减的来源:
1. Attention vs MLP: 哪个子层导致正确答案信号下降?
2. Per-head direct effect: 哪些 attention head 写入了错误信号?
3. Layer-by-layer logit attribution: 每层对最终 logits 的贡献

核心方法:
- 在每层 l, 分别采集: residual stream pre-attn, post-attn, post-mlp
- 用 logit lens 解码每个, 看 CIS 怎么变
- 如果 post-attn CIS < pre-attn CIS → attention 在抑制正确信号
- 如果 post-mlp CIS < post-attn CIS → MLP 在抑制
- per-head: 对每个 head, 计算它对 residual 的贡献, 再用 logit lens 解码
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


def get_attn_mlp_modules(model, layer_idx):
    """获取某一层的 attention 和 MLP 模块."""
    layer = model.model.layers[layer_idx]
    attn = None
    mlp = None
    # Qwen3: self_attn, mlp
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
    return attn, mlp, layer


def collect_component_hiddens(model, prompt_ids):
    """采集每层的 post_mlp (layer output), attn_out, mlp_out.

    策略 (不依赖 forward_pre_hook):
    - post_mlp: layer 的 forward hook 输出
    - attn_out: attn 子模块的 forward hook 输出
    - mlp_out: mlp 子模块的 forward hook 输出
    - pre_attn: 上一层的 post_mlp (或 embedding for layer 0)

    Qwen3 layer 结构:
        h = input
        attn_out = self_attn(input)
        h_mid = input + attn_out   (post_attn)
        mlp_out = mlp(norm(h_mid))
        h_out = h_mid + mlp_out    (post_mlp = layer output)
    """
    layers = get_residual_layers(model)
    num_layers = len(layers)
    device = model.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    components = {i: {"pre_attn": None, "attn_out": None, "post_attn": None,
                       "mlp_out": None, "post_mlp": None} for i in range(num_layers)}

    def make_post_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h_out = output[0]
            else:
                h_out = output
            components[idx]["post_mlp"] = h_out[0, -1, :].detach().clone()
        return hook

    def make_attn_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output
            components[idx]["attn_out"] = attn_out[0, -1, :].detach().clone()
        return hook

    def make_mlp_hook(idx):
        def hook(module, input, output):
            components[idx]["mlp_out"] = output[0, -1, :].detach().clone()
        return hook

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_post_hook(i)))
        attn, mlp, _ = get_attn_mlp_modules(model, i)
        if attn is not None:
            hooks.append(attn.register_forward_hook(make_attn_hook(i)))
        if mlp is not None:
            hooks.append(mlp.register_forward_hook(make_mlp_hook(i)))

    with torch.no_grad():
        model(input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    # 计算 pre_attn: 上一层的 post_mlp (或 embedding)
    # 用 post_mlp[l-1] 作为 pre_attn[l], post_mlp[-1] 用 embedding (近似用 post_mlp[0])
    for i in range(num_layers):
        if i == 0:
            # layer 0 的输入是 embedding, 用 attn_out 反推: pre_attn = post_mlp - attn_out - mlp_out
            if components[i]["post_mlp"] is not None and components[i]["attn_out"] is not None and components[i]["mlp_out"] is not None:
                components[i]["pre_attn"] = components[i]["post_mlp"] - components[i]["attn_out"] - components[i]["mlp_out"]
        else:
            # pre_attn[i] = post_mlp[i-1]
            if components[i-1]["post_mlp"] is not None:
                components[i]["pre_attn"] = components[i-1]["post_mlp"].clone()
        # post_attn = pre_attn + attn_out
        if components[i]["pre_attn"] is not None and components[i]["attn_out"] is not None:
            components[i]["post_attn"] = components[i]["pre_attn"] + components[i]["attn_out"]

    return components, num_layers


def logit_lens_cis(hidden, target_token, gen_token, final_norm, unembed):
    """对单个 hidden state 用 logit lens 算 CIS (用 float32)."""
    h = hidden.to(unembed.device).float()
    normed = final_norm(h)
    logits = F.linear(normed, unembed.float())
    log_probs = F.log_softmax(logits, dim=-1)
    if target_token >= logits.shape[-1] or gen_token >= logits.shape[-1]:
        return 0.0
    return log_probs[target_token].item() - log_probs[gen_token].item()


def run_component_decomposition(model, tokenizer, prepared, all_results, n_samples=50):
    """主实验: 分解 attention vs MLP 对 CIS 变化的贡献."""
    print("\n" + "=" * 70)
    print("P1: Component-level Decomposition")
    print("=" * 70)

    final_norm = model.model.norm
    unembed = model.lm_head.weight

    results = []
    for i, s in enumerate(tqdm(prepared[:n_samples], desc="Components")):
        try:
            result = next((r for r in all_results if r["id"] == s["id"]), None)
            if result is None:
                continue

            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            # 从 generated 文本重新 tokenize gen_token
            gen_text = result.get("generated", "")
            if gen_text:
                gen_full = tokenizer.encode(s["prompt_text"] + gen_text, add_special_tokens=False)
                gen_ids = gen_full[len(prompt_ids):]
                gen_token = gen_ids[0] if gen_ids else target_token
            else:
                gen_token = target_token

            components, num_layers = collect_component_hiddens(model, prompt_ids)

            # 每层: 计算各组件的 CIS
            layer_cis = []
            for l in range(num_layers):
                c = components[l]
                cis_pre = logit_lens_cis(c["pre_attn"], target_token, gen_token,
                                          final_norm, unembed) if c["pre_attn"] is not None else 0
                cis_post_attn = logit_lens_cis(c["post_attn"], target_token, gen_token,
                                                 final_norm, unembed) if c["post_attn"] is not None else 0
                cis_post_mlp = logit_lens_cis(c["post_mlp"], target_token, gen_token,
                                                final_norm, unembed) if c["post_mlp"] is not None else 0

                # 贡献
                attn_contrib = cis_post_attn - cis_pre      # attention 对 CIS 的改变
                mlp_contrib = cis_post_mlp - cis_post_attn   # MLP 对 CIS 的改变
                total_contrib = cis_post_mlp - cis_pre

                layer_cis.append({
                    "layer": l,
                    "cis_pre_attn": cis_pre,
                    "cis_post_attn": cis_post_attn,
                    "cis_post_mlp": cis_post_mlp,
                    "attn_contrib": attn_contrib,
                    "mlp_contrib": mlp_contrib,
                    "total_contrib": total_contrib,
                })

            results.append({
                "id": s["id"],
                "task": s["task"],
                "answer": s["answer"],
                "generated": result["generated"],
                "final_correct": result["final_correct"],
                "layer_cis": layer_cis,
            })

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # 分析
    print("\n" + "=" * 70)
    print("COMPONENT ANALYSIS")
    print("=" * 70)

    correct = [r for r in results if r["final_correct"]]
    incorrect = [r for r in results if not r["final_correct"]]

    for label, group in [("Correct", correct), ("Incorrect", incorrect)]:
        if not group:
            continue
        print(f"\n  {label} (n={len(group)}):")

        # 找 attention 贡献最负的层 (抑制正确信号)
        attn_contribs = np.array([[r["layer_cis"][l]["attn_contrib"] for l in range(len(r["layer_cis"]))]
                                    for r in group])
        mlp_contribs = np.array([[r["layer_cis"][l]["mlp_contrib"] for l in range(len(r["layer_cis"]))]
                                   for r in group])

        mean_attn = attn_contribs.mean(axis=0)
        mean_mlp = mlp_contribs.mean(axis=0)

        # 找 attention 抑制最强的层
        attn_suppress_layers = np.argsort(mean_attn)[:5]
        print(f"    Attention 抑制最强的层 (CIS 下降最多):")
        for l in attn_suppress_layers:
            print(f"      Layer {l}: attn_contrib={mean_attn[l]:.3f}")

        # 找 MLP 抑制最强的层
        mlp_suppress_layers = np.argsort(mean_mlp)[:5]
        print(f"    MLP 抑制最强的层:")
        for l in mlp_suppress_layers:
            print(f"      Layer {l}: mlp_contrib={mean_mlp[l]:.3f}")

        # 总贡献
        total_attn = mean_attn.sum()
        total_mlp = mean_mlp.sum()
        print(f"    总 attention 贡献: {total_attn:.3f}")
        print(f"    总 MLP 贡献: {total_mlp:.3f}")
        if total_attn < total_mlp:
            print(f"    → Attention 是主要抑制源")
        else:
            print(f"    → MLP 是主要抑制源")

    return results


def run_per_head_analysis(model, tokenizer, prepared, all_results, n_samples=20):
    """Per-head direct effect: 每个 attention head 对 CIS 的贡献.

    方法: 对每层每个 head, 计算它的输出对 residual stream 的贡献,
    然后用 logit lens 看 CIS.
    """
    print("\n" + "=" * 70)
    print("P1: Per-Head Direct Effect")
    print("=" * 70)

    final_norm = model.model.norm
    unembed = model.lm_head.weight
    layers = get_residual_layers(model)
    num_layers = len(layers)
    device = model.device

    # 只看最后 6 层 (信号衰减的关键层)
    target_layers = list(range(max(0, num_layers-6), num_layers))

    results = []
    for i, s in enumerate(tqdm(prepared[:n_samples], desc="Per-head")):
        try:
            result = next((r for r in all_results if r["id"] == s["id"]), None)
            if result is None:
                continue

            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            gen_token = result.get("generated_token_ids", [target_token])[0]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # 获取 attention weights 和 head 输出
            # Qwen3: attn output = o_proj(attn_weights @ v)
            # per-head: head_i 的贡献 = o_proj[:, i*dh:(i+1)*dh] @ (attn_weights[i] @ v[i])

            head_results = {}
            with torch.no_grad():
                # 注册 hook 采集 attention 的内部状态
                attn_internals = {}

                def make_attn_internals_hook(layer_idx):
                    def hook(module, input, output):
                        # 保存 attention weights
                        if hasattr(module, 'num_heads'):
                            attn_internals[layer_idx] = {
                                'output': output[0] if isinstance(output, tuple) else output
                            }
                    return hook

                hooks = []
                for l in target_layers:
                    attn, _, _ = get_attn_mlp_modules(model, l)
                    if attn is not None:
                        hooks.append(attn.register_forward_hook(make_attn_internals_hook(l)))

                model(input_ids, use_cache=False)
                for h in hooks:
                    h.remove()

            # 简化版: 直接用 attention output 的 norm 作为 head 影响力的 proxy
            # 严格版需要拆解 o_proj, 这里先做简化
            for l in target_layers:
                attn, _, _ = get_attn_mlp_modules(model, l)
                if attn is None or l not in attn_internals:
                    continue

                attn_out = attn_internals[l]['output'][0, -1, :]  # [hidden]
                # 用 logit lens 看 attn_out 本身的 CIS 贡献
                h = attn_out.to(unembed.device).to(unembed.dtype)
                normed = final_norm(h)
                logits = F.linear(normed, unembed)
                log_probs = F.log_softmax(logits, dim=-1)
                if target_token < logits.shape[-1] and gen_token < logits.shape[-1]:
                    cis_attn = log_probs[target_token].item() - log_probs[gen_token].item()
                    head_results[l] = {"attn_out_cis": cis_attn}

            results.append({
                "id": s["id"],
                "final_correct": result["final_correct"],
                "head_results": head_results,
            })

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    # 分析
    print("\n  Per-head analysis (simplified: attn output CIS per layer):")
    for label, group in [("Correct", [r for r in results if r["final_correct"]]),
                           ("Incorrect", [r for r in results if not r["final_correct"]])]:
        if not group:
            continue
        print(f"\n  {label} (n={len(group)}):")
        for l in target_layers:
            ciss = [r["head_results"].get(l, {}).get("attn_out_cis", 0) for r in group
                    if l in r["head_results"]]
            if ciss:
                print(f"    Layer {l}: attn_out CIS = {np.mean(ciss):.3f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--n_heads", type=int, default=20)
    args = parser.parse_args()

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

    component_results = run_component_decomposition(model, tokenizer, prepared, all_results, n_samples=args.n_components)
    head_results = run_per_head_analysis(model, tokenizer, prepared, all_results, n_samples=args.n_heads)

    out = {"component_decomposition": component_results,
            "per_head_analysis": head_results}
    out_file = os.path.join(config.DATA_DIR, "p1_components_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
