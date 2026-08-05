"""
MLP causal intervention (layer-35 patching).

审稿人批评: DLA 只是归因, 不是因果. 需要证明
  MLP intervention → ΔCIS > 0 → Δaccuracy > 0.

实验:
1. 对错误样本, 采集它自己的 (corrupted) layer-35 MLP 输出
2. 用另一个正确样本的 layer-35 MLP 输出替换 (或在该层做 patching)
3. 看最终 gold 正确率是否恢复
4. 对照: 换随机样本的 MLP 输出 / 不做干预

注意: 需要同一 prompt 的 clean 和 corrupted run.
这里用简化版: 对每个错误样本, 从正确样本池里找同 task 的样本,
替换其 layer-35 MLP 输出, 看能否提高该样本 gold token 的 log-probability.

用法: python run_mlp_patch.py --model qwen
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


def get_final_norm_and_unembed(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm, model.lm_head.weight
    raise ValueError


@torch.no_grad()
def collect_layer_mlp_outputs(model, prompt_ids, mlp_layer):
    """采集某一层 MLP 的输出 hidden state (目标位置)."""
    layers = get_residual_layers(model)
    device = model.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    mlp_out = {}
    attn = getattr(layers[mlp_layer], "self_attn", None)
    mlp = getattr(layers[mlp_layer], "mlp", None)

    def make_mlp_hook():
        def hook(module, input, output):
            mlp_out["out"] = output[0, -1, :].detach().clone()
        return hook

    def make_attn_hook():
        def hook(module, input, output):
            mlp_out["attn"] = output[0, -1, :].detach().clone()
        return hook

    hooks = []
    if mlp is not None:
        hooks.append(mlp.register_forward_hook(make_mlp_hook()))
    if attn is not None:
        hooks.append(attn.register_forward_hook(make_attn_hook()))

    model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    return mlp_out


@torch.no_grad()
def patch_mlp_generate(model, tokenizer, prompt_ids, gold_token, donor_mlp_out, mlp_layer):
    """在 mlp_layer 用 donor 的 MLP 输出替换, 然后生成."""
    layers = get_residual_layers(model)
    device = model.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = ()
        # 替换目标位置 (最后一个 token) 的 MLP 输出
        h[0, -1, :] = donor_mlp_out.to(h.device).to(h.dtype)
        if rest:
            return (h,) + rest
        return h

    mlp = getattr(layers[mlp_layer], "mlp", None)
    hook = mlp.register_forward_hook(patch_hook)
    out = model.generate(
        input_ids, max_new_tokens=32, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    hook.remove()
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def baseline_logprob(model, prompt_ids, gold_token, final_norm, unembed):
    """不干预时的最终层 gold log-probability."""
    device = model.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(input_ids, use_cache=False)
    logits = out.logits[0, -1, :]
    return F.log_softmax(logits, dim=-1)[gold_token].item()


@torch.no_grad()
def patch_mlp_logprob(model, prompt_ids, gold_token, donor_mlp_out, mlp_layer, final_norm, unembed):
    """在 mlp_layer 用 donor 替换后, 计算最终层 gold log-probability."""
    layers = get_residual_layers(model)
    device = model.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = ()
        h[0, -1, :] = donor_mlp_out.to(h.device).to(h.dtype)
        if rest:
            return (h,) + rest
        return h

    mlp = getattr(layers[mlp_layer], "mlp", None)
    hook = mlp.register_forward_hook(patch_hook)
    out = model(input_ids, use_cache=False)
    hook.remove()

    logits = out.logits[0, -1, :]
    lp = F.log_softmax(logits, dim=-1)
    return lp[gold_token].item()


def run_mlp_patch(model_key, mlp_layer=None, n_samples=50):
    cfg = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"MLP Causal Intervention: {cfg['name']}")
    print(f"{'='*70}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    if mlp_layer is None:
        mlp_layer = num_layers - 1  # 最后一层
    final_norm, unembed = get_final_norm_and_unembed(model)
    device = model.device

    # 加载数据
    samples = load_data(tokenizer, model_key, n_per_task=50)

    # 分离正确/错误
    correct_samples = []
    incorrect_samples = []
    for s in tqdm(samples, desc="Classify"):
        gen = model.generate(
            torch.tensor([s["prompt_ids"]], dtype=torch.long, device=device),
            max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0][len(s["prompt_ids"]):], skip_special_tokens=True).strip()
        if is_answer_correct(text, s["aliases"]):
            correct_samples.append(s)
        else:
            incorrect_samples.append(s)

    print(f"Correct: {len(correct_samples)}, Incorrect: {len(incorrect_samples)}")

    # 对错误样本, 用同 task 的正确样本 MLP 输出替换
    results = []
    correct_by_task = {}
    for c in correct_samples:
        correct_by_task.setdefault(c.get("task", "unknown"), []).append(c)

    for s in tqdm(incorrect_samples[:n_samples], desc="MLP patch"):
        gold_token = s["primary_answer_ids"][0]
        gold_aliases = s["aliases"]
        task = s.get("task", "unknown")

        # 基线 (不 patch) gold log-prob
        base_lp = baseline_logprob(model, s["prompt_ids"], gold_token, final_norm, unembed)

        # 找同 task 的正确样本作为 donor
        donors = correct_by_task.get(task, [])
        if not donors:
            continue
        donor = donors[np.random.randint(len(donors))]
        donor_mlp_out = collect_layer_mlp_outputs(model, donor["prompt_ids"], mlp_layer)["out"]

        # patch 后 gold log-prob
        patched_lp = patch_mlp_logprob(model, s["prompt_ids"], gold_token,
                                       donor_mlp_out, mlp_layer, final_norm, unembed)

        # patch 后生成
        patched_gen = patch_mlp_generate(model, tokenizer, s["prompt_ids"],
                                         gold_token, donor_mlp_out, mlp_layer)
        patched_correct = is_answer_correct(patched_gen, gold_aliases)

        results.append({
            "id": s["id"],
            "answer": s["answer"],
            "base_gold_lp": base_lp,
            "patched_gold_lp": patched_lp,
            "delta_lp": patched_lp - base_lp,
            "patched_correct": patched_correct,
        })

    # 分析
    print("\n--- 结果 ---")
    n = len(results)
    if n == 0:
        print("  ! 没有可用结果")
        return results

    deltas = [r["delta_lp"] for r in results]
    positive_delta = sum(1 for d in deltas if d > 0)
    recovered = sum(1 for r in results if r["patched_correct"])

    print(f"样本数: {n}")
    print(f"delta gold log-prob: mean={np.mean(deltas):.3f}, "
          f"positive={positive_delta}/{n} ({100*positive_delta/n:.1f}%)")
    print(f"patched 后答对: {recovered}/{n} ({100*recovered/n:.1f}%)")

    if np.mean(deltas) > 0 and positive_delta > 0.5 * n:
        print(f"\n✓ MLP patching 提高了 gold log-probability")
        print(f"  → MLP 干预对正确信号有因果贡献 (ΔCIS → Δlog-prob)")
    else:
        print(f"\n? MLP patching 未显著提高 gold log-probability")

    out_file = os.path.join(config.DATA_DIR, f"mlp_patch_{model_key}.json")
    with open(out_file, "w") as f:
        json.dump({"model": cfg["name"], "mlp_layer": mlp_layer, "results": results}, f, indent=2)
    print(f"Saved: {out_file}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    run_mlp_patch(args.model, mlp_layer=args.layer, n_samples=args.n)
