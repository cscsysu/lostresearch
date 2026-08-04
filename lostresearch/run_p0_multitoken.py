"""
P0 实验 1: Multi-token sequence-level CIS

用 teacher forcing 算完整答案序列的 log-prob, 而非只算首 token.

对每个样本, 在每层 l:
  CIS_seq_l = (1/T) * sum_t log P_l(y*_t | x, y*_<t)
            - (1/S) * sum_t log P_l(y^_t | x, y^_<t)

其中 y* 是 gold answer, y^ 是 generated answer.
需要把 prompt + answer 一起 forward, 在每个位置 i 用层 l 的 hidden 预测 token i+1.
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


def collect_sequence_hiddens(model, prompt_ids, answer_ids):
    """跑 prompt + answer 的 forward, 采集每层每个位置的 hidden state.

    用于 teacher forcing:
    - 位置 i 的 hidden state 预测 token i+1
    - 对 gold answer: 用 prompt_ids + answer_ids, 在 prompt 末尾开始算 answer 的 log-prob
    - 对 generated answer: 用 prompt_ids + gen_ids
    """
    layers = get_residual_layers(model)
    num_layers = len(layers)
    device = model.device
    full_ids = prompt_ids + answer_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    hiddens = {}
    hooks = []

    def make_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hiddens[idx] = h[0].detach().clone()  # [seq, hidden]
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    return hiddens, num_layers


def compute_sequence_logprob_at_layer(hidden_seq, target_ids, prompt_len,
                                         final_norm, unembed):
    """在某一层, 用 teacher forcing 算 target_ids 的序列 log-prob.

    位置 i 的 hidden state 预测 token i+1.
    target_ids[t] 由位置 prompt_len-1+t 的 hidden 预测.
    """
    log_probs_sum = 0.0
    for i, target_token in enumerate(target_ids):
        pos = prompt_len - 1 + i  # 位置 pos 的 hidden 预测 pos+1 = target_ids[i]
        if pos < 0 or pos >= hidden_seq.shape[0]:
            return -1e9
        h = hidden_seq[pos].to(unembed.device).to(unembed.dtype)
        normed = final_norm(h)
        logits = F.linear(normed, unembed)
        log_probs = F.log_softmax(logits, dim=-1)
        if target_token >= logits.shape[-1]:
            return -1e9
        log_probs_sum += log_probs[target_token].item()
    return log_probs_sum / max(len(target_ids), 1)  # 平均 per-token


def run_multitoken_cis(model, tokenizer, prepared, all_results, n_samples=200):
    """计算 multi-token sequence-level CIS."""
    print("\n" + "=" * 70)
    print("P0-1: Multi-token Sequence-level CIS")
    print("=" * 70)

    final_norm = model.model.norm
    unembed = model.lm_head.weight
    device = model.device

    results = []
    skipped = 0
    for i, s in enumerate(tqdm(prepared[:n_samples], desc="Multi-token CIS")):
        try:
            result = next((r for r in all_results if r["id"] == s["id"]), None)
            if result is None:
                continue

            prompt_ids = s["prompt_ids"]
            gold_ids = s["primary_answer_ids"]
            gen_ids = result.get("generated_token_ids", gold_ids)

            if len(gold_ids) <= 1 or len(gen_ids) <= 1:
                skipped += 1
                continue

            # 1. Gold sequence forward (prompt + gold answer)
            gold_hiddens, num_layers = collect_sequence_hiddens(model, prompt_ids, gold_ids)
            gold_prompt_len = len(prompt_ids)

            # 2. Generated sequence forward (prompt + generated answer)
            gen_hiddens, _ = collect_sequence_hiddens(model, prompt_ids, gen_ids)
            gen_prompt_len = len(prompt_ids)

            # 3. 每层计算 sequence-level CIS
            cis_seq_per_layer = []
            cis_first_token_per_layer = []  # 对比: 只用首 token

            for l in range(num_layers):
                # Sequence-level
                lp_gold_seq = compute_sequence_logprob_at_layer(
                    gold_hiddens[l], gold_ids, gold_prompt_len, final_norm, unembed)
                lp_gen_seq = compute_sequence_logprob_at_layer(
                    gen_hiddens[l], gen_ids, gen_prompt_len, final_norm, unembed)
                cis_seq = lp_gold_seq - lp_gen_seq
                cis_seq_per_layer.append(cis_seq)

                # First-token only (对比): 用 prompt 最后位置的 hidden
                h_gold = gold_hiddens[l][gold_prompt_len - 1].to(unembed.device).to(unembed.dtype)
                h_gen = gen_hiddens[l][gen_prompt_len - 1].to(unembed.device).to(unembed.dtype)
                lp_gold_first = F.log_softmax(F.linear(final_norm(h_gold), unembed), dim=-1)[gold_ids[0]].item()
                lp_gen_first = F.log_softmax(F.linear(final_norm(h_gen), unembed), dim=-1)[gen_ids[0]].item()
                cis_first_token_per_layer.append(lp_gold_first - lp_gen_first)

            results.append({
                "id": s["id"],
                "task": s["task"],
                "answer": s["answer"],
                "generated": result["generated"],
                "gold_token_len": len(gold_ids),
                "gen_token_len": len(gen_ids),
                "final_correct": result["final_correct"],
                "cis_seq": cis_seq_per_layer,
                "cis_first_token": cis_first_token_per_layer,
            })

        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            continue

    print(f"\n  Computed: {len(results)}, Skipped (single token): {skipped}")

    # 分析: 对比 sequence-level vs first-token
    print("\n  Comparison: Sequence-level vs First-token CIS")
    print(f"  {'Metric':<30} {'Seq-level':<12} {'First-token':<12}")
    print("  " + "-" * 54)

    correct = [r for r in results if r["final_correct"]]
    incorrect = [r for r in results if not r["final_correct"]]

    for label, group in [("Correct", correct), ("Incorrect", incorrect)]:
        if not group:
            continue
        seq_peaks = [max(r["cis_seq"]) for r in group]
        first_peaks = [max(r["cis_first_token"]) for r in group]
        seq_finals = [r["cis_seq"][-1] for r in group]
        first_finals = [r["cis_first_token"][-1] for r in group]

        print(f"  {label} (n={len(group)}):")
        print(f"    {'Peak CIS':<28} {np.mean(seq_peaks):<12.3f} {np.mean(first_peaks):<12.3f}")
        print(f"    {'Final CIS':<28} {np.mean(seq_finals):<12.3f} {np.mean(first_finals):<12.3f}")

    # 相关性
    seq_arr = np.array([r["cis_seq"][-1] for r in results])
    first_arr = np.array([r["cis_first_token"][-1] for r in results])
    corr = np.corrcoef(seq_arr, first_arr)[0, 1]
    print(f"\n  Final CIS correlation (seq vs first-token): {corr:.3f}")
    if corr > 0.8:
        print("  ✓ Sequence-level 与 first-token 高度相关 → first-token 近似合理")
    elif corr > 0.5:
        print(f"  ? 中等相关 → multi-token 提供额外信息")
    else:
        print("  ! 低相关 → multi-token 与 first-token 给出不同结论")

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

    results = run_multitoken_cis(model, tokenizer, prepared, all_results, n_samples=200)

    out_file = os.path.join(config.DATA_DIR, "multitoken_cis_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
