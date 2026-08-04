"""
P0 实验 4: Tuned Lens 对照

训练每层 affine translator, 验证 CIS 轨迹不是 raw LM head 的 artifact.

Tuned Lens:
  对每层 l, 训练 (A_l, b_l) 使得 A_l * h_l + b_l 经过 final_norm + unembedding
  后尽量接近最终层的 logits.

用 1000 题的 80% 训练 translator, 20% 测试.
然后用 tuned lens 重算 CIS, 和 raw logit lens 对比.
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


def collect_all_hiddens(model, samples, n_samples=200):
    """采集 n_samples 个样本的所有层 hidden state + 最终层 logits."""
    print(f"Collecting hiddens for {n_samples} samples...")
    layers = get_residual_layers(model)
    num_layers = len(layers)
    device = model.device

    all_hiddens = {i: [] for i in range(num_layers)}
    all_final_logits = []

    hooks = []
    hidden_buffer = {i: None for i in range(num_layers)}

    def make_hook(idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_buffer[idx] = h[0, -1, :].detach().clone()  # last token, [hidden]
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    for i, s in enumerate(tqdm(samples[:n_samples])):
        prompt_ids = s["prompt_ids"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
        final_logits = out.logits[0, -1, :].cpu()  # [vocab]
        all_final_logits.append(final_logits)

        for l in range(num_layers):
            all_hiddens[l].append(hidden_buffer[l].cpu())

    for h in hooks:
        h.remove()

    # Stack
    for l in range(num_layers):
        all_hiddens[l] = torch.stack(all_hiddens[l])  # [n, hidden]
    all_final_logits = torch.stack(all_final_logits)  # [n, vocab]

    return all_hiddens, all_final_logits, num_layers


def train_tuned_lens(model, hiddens, final_logits, num_layers, epochs=500, lr=1e-3):
    """训练每层的 affine translator (A_l, b_l).

    目标: A_l * h_l + b_l 经过 final_norm + unembedding 后接近 final_logits.
    简化: 直接学一个 [hidden, vocab] 的线性映射, 不经过 final_norm.
    """
    print("Training tuned lens translators...")
    device = model.device

    final_norm = model.model.norm
    unembed = model.lm_head.weight  # [vocab, hidden]

    translators = {}
    for l in range(num_layers):
        h = hiddens[l].to(device)  # [n, hidden]
        y = final_logits.to(device)  # [n, vocab]

        # 学一个 affine: A (hidden, hidden) + b (hidden)
        # 然后通过 final_norm + unembedding
        A = torch.eye(h.shape[1], device=device, dtype=torch.float32)
        b = torch.zeros(h.shape[1], device=device, dtype=torch.float32)
        A.requires_grad_(True)
        b.requires_grad_(True)

        optimizer = torch.optim.Adam([A, b], lr=lr)
        h_f = h.float()
        y_f = y.float()
        unembed_f = unembed.float()

        for epoch in range(epochs):
            optimizer.zero_grad()
            translated = h_f @ A.T + b  # [n, hidden]
            # 通过 final_norm + unembedding (用 float32 避免 dtype 冲突)
            normed = final_norm(translated)
            logits = F.linear(normed, unembed_f)  # [n, vocab]
            loss = F.mse_loss(logits, y_f)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"  Layer {l}: epoch {epoch}, loss={loss.item():.4f}")

        translators[l] = {"A": A.detach().cpu(), "b": b.detach().cpu()}

    return translators


def compute_cis_with_tuned_lens(model, tokenizer, translators, samples, n_samples=200):
    """用 tuned lens 重算 CIS 轨迹."""
    print("Computing CIS with tuned lens...")
    layers = get_residual_layers(model)
    num_layers = len(layers)
    device = model.device
    final_norm = model.model.norm
    unembed = model.lm_head.weight
    unembed_f = unembed.float()  # 用 float32 避免 dtype 冲突

    results = []
    for i, s in enumerate(tqdm(samples[:n_samples])):
        prompt_ids = s["prompt_ids"]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 采集每层 hidden
        hidden_buffer = {}
        def make_hook(idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                hidden_buffer[idx] = h[0, -1, :].detach().clone()
            return hook

        hooks = [layer.register_forward_hook(make_hook(l)) for l, layer in enumerate(layers)]
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
        for h in hooks:
            h.remove()

        # 用 tuned lens 解码
        cis_tuned = []
        cis_raw = []
        correct_rank_tuned = []
        correct_rank_raw = []

        # 从 generated 文本重新 tokenize gen_token
        gen_text = s.get("generated", "")
        target_token = s["primary_answer_ids"][0]

        if gen_text:
            # 用 prompt_ids 的实际 token 来对齐
            prompt_ids = s["prompt_ids"]
            # decode prompt_ids 回文本再拼接
            prompt_decoded = tokenizer.decode(prompt_ids)
            gen_full_ids = tokenizer.encode(prompt_decoded + gen_text, add_special_tokens=False)
            gen_ids = gen_full_ids[len(prompt_ids):]
            gen_token = gen_ids[0] if gen_ids else target_token
        else:
            gen_token = target_token

        if i == 0:
            print(f"  [diag] sample {s['id']}: target={target_token}, gen_token={gen_token}, "
                  f"gen_text='{gen_text[:30]}', match={target_token == gen_token}")

        for l in range(num_layers):
            h = hidden_buffer[l].to(device).float()

            # Raw logit lens (用 float32)
            normed_raw = final_norm(h)
            logits_raw = F.linear(normed_raw, unembed_f)
            log_probs_raw = F.log_softmax(logits_raw, dim=-1)
            lp_correct_raw = log_probs_raw[target_token].item()
            lp_gen_raw = log_probs_raw[gen_token].item()
            cis_raw.append(lp_correct_raw - lp_gen_raw)
            # rank
            sorted_raw = torch.argsort(logits_raw, descending=True)
            rank_raw = (sorted_raw == target_token).nonzero(as_tuple=True)[0].item()
            correct_rank_raw.append(rank_raw)

            # Tuned lens
            A = translators[l]["A"].to(device).float()
            b = translators[l]["b"].to(device).float()
            translated = A @ h + b
            normed_tuned = final_norm(translated)
            logits_tuned = F.linear(normed_tuned, unembed_f)
            log_probs_tuned = F.log_softmax(logits_tuned, dim=-1)
            lp_correct_tuned = log_probs_tuned[target_token].item()
            lp_gen_tuned = log_probs_tuned[gen_token].item()
            cis_tuned.append(lp_correct_tuned - lp_gen_tuned)
            # rank
            sorted_tuned = torch.argsort(logits_tuned, descending=True)
            rank_tuned = (sorted_tuned == target_token).nonzero(as_tuple=True)[0].item()
            correct_rank_tuned.append(rank_tuned)

        results.append({
            "id": s["id"],
            "cis_raw": cis_raw,
            "cis_tuned": cis_tuned,
            "correct_rank_raw": correct_rank_raw,
            "correct_rank_tuned": correct_rank_tuned,
            "final_correct": s.get("final_correct", False),
        })

    return results


def compare_raw_vs_tuned(results):
    """对比 raw logit lens 和 tuned lens 的轨迹 + 重算 8.5%."""
    print("\n" + "=" * 70)
    print("P0-4: Tuned Lens vs Raw Logit Lens")
    print("=" * 70)

    correct_samples = [r for r in results if r["final_correct"]]
    incorrect_samples = [r for r in results if not r["final_correct"]]

    for label, group in [("Correct", correct_samples), ("Incorrect", incorrect_samples)]:
        if not group:
            continue
        print(f"\n  {label} (n={len(group)}):")

        for lens_name, cis_key, rank_key in [("raw", "cis_raw", "correct_rank_raw"),
                                               ("tuned", "cis_tuned", "correct_rank_tuned")]:
            sign_change = sum(1 for r in group
                             if any(r[cis_key][i] > 0 and r[cis_key][i+1] < 0
                                    for i in range(len(r[cis_key])-1)))
            peaks = [max(r[cis_key]) for r in group]
            finals = [r[cis_key][-1] for r in group]
            deltas = [max(r[cis_key][1:-1]) - r[cis_key][-1] for r in group
                      if len(r[cis_key]) > 2]
            # peak rank (排除最终层)
            peak_ranks = [min(r[rank_key][1:-1]) for r in group if len(r[rank_key]) > 2]

            print(f"    {lens_name}: sign_change={sign_change}/{len(group)}, "
                  f"peak_CIS={np.mean(peaks):.2f}, final_CIS={np.mean(finals):.2f}, "
                  f"mid-final_delta={np.mean(deltas):.2f}")
            print(f"           peak_rank (excl final): median={np.median(peak_ranks):.0f}")

    # 峰值层一致性
    consistent = 0
    total = 0
    for r in results:
        raw = r["cis_raw"]
        tuned = r["cis_tuned"]
        if len(raw) == len(tuned) and len(raw) > 2:
            raw_peak_layer = raw.index(max(raw))
            tuned_peak_layer = tuned.index(max(tuned))
            if abs(raw_peak_layer - tuned_peak_layer) <= 2:
                consistent += 1
            total += 1

    if total > 0:
        print(f"\n  峰值层一致性 (±2层): {consistent}/{total} ({100*consistent/total:.1f}%)")

    # === 关键: 在 tuned lens 下重算 8.5% ===
    print(f"\n  --- Competitive-Decay Rate (rank≤5 AND CIS>0 AND final CIS<0) ---")
    print(f"  {'Lens':<10} {'Count':<10} {'Rate':<10} {'95% CI'}")
    print("  " + "-" * 45)

    for lens_name, cis_key, rank_key in [("raw", "cis_raw", "correct_rank_raw"),
                                           ("tuned", "cis_tuned", "correct_rank_tuned")]:
        count = 0
        for r in incorrect_samples:
            cis = r[cis_key]
            ranks = r[rank_key]
            if len(cis) < 4 or len(ranks) < 4:
                continue
            mid_cis = cis[1:-1]
            mid_ranks = ranks[1:-1]
            has_competitive = any(
                mid_cis[i] > 0 and mid_ranks[i] <= 5
                for i in range(len(mid_cis))
            )
            if has_competitive and cis[-1] < 0:
                count += 1

        pct = 100 * count / len(incorrect_samples) if incorrect_samples else 0
        # Bootstrap CI
        boots = []
        for _ in range(1000):
            sample = np.random.choice(incorrect_samples, len(incorrect_samples), replace=True)
            cnt = sum(1 for r in sample
                      if any(c > 0 and rk <= 5 for c, rk in zip(r[cis_key][1:-1], r[rank_key][1:-1]))
                      and r[cis_key][-1] < 0)
            boots.append(100 * cnt / len(sample))
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        print(f"  {lens_name:<10} {count}/{len(incorrect_samples):<8} {pct:.1f}%     [{ci_low:.1f}, {ci_high:.1f}]")

    # 判断
    raw_count = sum(1 for r in incorrect_samples
                    if any(c > 0 and rk <= 5 for c, rk in zip(r["cis_raw"][1:-1], r["correct_rank_raw"][1:-1]))
                    and r["cis_raw"][-1] < 0)
    tuned_count = sum(1 for r in incorrect_samples
                      if any(c > 0 and rk <= 5 for c, rk in zip(r["cis_tuned"][1:-1], r["correct_rank_tuned"][1:-1]))
                      and r["cis_tuned"][-1] < 0)
    print(f"\n  判断:")
    if tuned_count > 0:
        ratio = raw_count / tuned_count if tuned_count > 0 else 0
        print(f"  Raw: {raw_count}, Tuned: {tuned_count}, ratio: {ratio:.1f}x")
        if ratio < 2:
            print(f"  ✓ Tuned lens 下 8.5% 仍然存在 (ratio < 2x)")
        else:
            print(f"  ? Raw 夸大了 {ratio:.1f}x, tuned 下比例显著降低")
    else:
        print(f"  ✗ Tuned lens 下 competitive-decay 为 0 → 核心结论可能不成立")


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

    # 加载已有结果 (为了拿 generated 文本和 final_correct)
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            existing = {r["id"]: r for r in json.load(f)}
        for s in prepared:
            if s["id"] in existing:
                s["final_correct"] = existing[s["id"]]["final_correct"]
                s["generated"] = existing[s["id"]].get("generated", "")

    # 1. 采集 hiddens + final logits
    hiddens, final_logits, num_layers = collect_all_hiddens(model, prepared, n_samples=200)

    # 2. 训练 tuned lens (或加载已保存的)
    translator_file = os.path.join(config.DATA_DIR, "tuned_lens_translators_Qwen3-8B.pt")
    if os.path.exists(translator_file):
        print(f"Loading saved translators from {translator_file}...")
        translators = torch.load(translator_file, map_location="cpu")
    else:
        translators = train_tuned_lens(model, hiddens, final_logits, num_layers, epochs=300)
        torch.save(translators, translator_file)
        print(f"Saved translators to {translator_file}")

    # 3. 用 tuned lens 重算 CIS
    results = compute_cis_with_tuned_lens(model, tokenizer, translators, prepared, n_samples=200)

    # 4. 对比
    compare_raw_vs_tuned(results)

    # 保存
    out_file = os.path.join(config.DATA_DIR, "tuned_lens_comparison_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
