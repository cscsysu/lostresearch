"""
Review v5: 区分 CISgen 和 CIScomp + per-model 分解

CISgen = log P(gold) - log P(generated)  [仅错误样本]
CIScomp = log P(gold) - log P(strongest competitor)  [正确+错误样本]

competitor = argmax_{y≠gold} P_final(y)

这个脚本不需要模型, 只用已保存的 top5 数据.
但 top5 只有 5 个 token, 不是完整 argmax.
近似: 用 top5 里第一个非 gold token 作为 competitor.
"""
import json
import os
import numpy as np

import config


def find_competitor_token(top5_final, target_token):
    """从最终层 top5 里找第一个非 gold 的 token 作为 competitor."""
    for item in top5_final:
        tid = item[0] if isinstance(item, list) else item
        if tid != target_token:
            return tid
    return None


def analyze_cis_comp(all_results):
    """分析 CIScomp (gold vs competitor) vs CISgen (gold vs generated)."""
    print("\n" + "=" * 70)
    print("Review v5: CISgen vs CIScomp")
    print("=" * 70)

    correct = [s for s in all_results if s["final_correct"]]
    incorrect = [s for s in all_results if not s["final_correct"]]

    print(f"\n正确样本: {len(correct)}, 错误样本: {len(incorrect)}")

    # === 检查正确样本的 CIS ===
    print("\n--- 正确样本 CIS 检查 ---")
    cis_finals = [s["cis"][-1] for s in correct if s.get("cis")]
    zero_count = sum(1 for c in cis_finals if abs(c) < 0.01)
    print(f"  CIS ≈ 0 (|CIS|<0.01): {zero_count}/{len(cis_finals)} ({100*zero_count/len(cis_finals):.1f}%)")
    print(f"  CIS < 0: {sum(1 for c in cis_finals if c < -0.01)}/{len(cis_finals)}")
    print(f"  → {100*zero_count/len(cis_finals):.1f}% 的正确样本 CIS≈0 (generated==gold)")
    print(f"  → 其余 {100*(len(cis_finals)-zero_count)/len(cis_finals):.1f}% 因 tokenization 差异 CIS≠0")

    # === CIScomp: 正确样本用 competitor ===
    # 但我们没有保存完整 logits, 只有 top5
    # 用 top5[最终层] 里第一个非 gold token 近似 competitor
    print("\n--- CIScomp (gold vs competitor, 近似) ---")
    print("注意: 用 top5 近似 argmax_{y≠gold}, 不完全准确")
    print()

    for label, group in [("Correct", correct), ("Incorrect", incorrect)]:
        cis_comp_finals = []
        cis_gen_finals = []

        for s in group:
            top5 = s.get("top5", [])
            cis = s.get("cis", [])
            if not top5 or not cis:
                continue

            # 最终层 top5
            top5_final = top5[-1] if top5 else []
            target_token = s.get("primary_answer_ids", [0])
            if isinstance(target_token, list):
                target_token = target_token[0] if target_token else 0

            competitor = find_competitor_token(top5_final, target_token)
            if competitor is None:
                continue

            # CISgen (gold - generated) = 已有的 cis
            cis_gen_finals.append(cis[-1])

            # CIScomp (gold - competitor): 从 top5 里算
            # top5 格式: [[token_id, prob], ...]
            # 找 gold 和 competitor 的 prob
            gold_prob = 0
            comp_prob = 0
            for item in top5_final:
                tid = item[0] if isinstance(item, list) else item
                prob = item[1] if isinstance(item, list) else 0
                if tid == target_token:
                    gold_prob = prob
                elif tid == competitor:
                    comp_prob = prob

            if gold_prob > 0 and comp_prob > 0:
                cis_comp = np.log(gold_prob) - np.log(comp_prob)
                cis_comp_finals.append(cis_comp)

        if cis_gen_finals and cis_comp_finals:
            print(f"  {label} (n={len(cis_comp_finals)}):")
            print(f"    CISgen (gold vs generated): mean={np.mean(cis_gen_finals):.3f}, median={np.median(cis_gen_finals):.3f}")
            print(f"    CIScomp (gold vs competitor): mean={np.mean(cis_comp_finals):.3f}, median={np.median(cis_comp_finals):.3f}")

    # === Per-model 8.9% 分解 ===
    print("\n--- 8.9% Per-Model 分解 ---")
    # 8.9% 是在 200 题子集上算的 (tuned lens)
    # 检查 per-task
    for task in ["triviaqa", "hotpotqa", "gsm8k"]:
        task_incorrect = [s for s in incorrect if s.get("task") == task]
        if not task_incorrect:
            continue
        count = sum(1 for s in task_incorrect
                    if any(c > 0 and r <= 5 for c, r in zip(s["cis"][1:-1], s["correct_rank"][1:-1]))
                    and s["cis"][-1] < 0)
        print(f"  {task}: {count}/{len(task_incorrect)} ({100*count/len(task_incorrect):.1f}%)")

    # === Per-model peak rank (三模型) ===
    print("\n--- Per-Model Peak Rank (排除最终层) ---")
    # Qwen
    c_peaks = [min(s["correct_rank"][1:-1]) for s in correct if len(s["correct_rank"]) > 2]
    i_peaks = [min(s["correct_rank"][1:-1]) for s in incorrect if len(s["correct_rank"]) > 2]
    c_top5 = sum(1 for r in c_peaks if r <= 4)  # 0-indexed, top-5 = rank 0-4
    i_top5 = sum(1 for r in i_peaks if r <= 4)
    print(f"  Qwen3-8B (n={len(correct)+len(incorrect)}):")
    print(f"    Correct: peak rank median={np.median(c_peaks):.0f}, top-5 rate={100*c_top5/len(c_peaks):.1f}%")
    print(f"    Incorrect: peak rank median={np.median(i_peaks):.0f}, top-5 rate={100*i_top5/len(i_peaks):.1f}%")

    # Llama / Mistral
    for model_key in ["llama", "mistral"]:
        f = os.path.join(config.DATA_DIR, f"cross_model_{model_key}.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            data = json.load(fh)
        results = data.get("trajectory_results", [])
        c = [r for r in results if r["final_correct"]]
        i = [r for r in results if not r["final_correct"]]
        if c and i:
            c_p = [min(r["correct_rank"][1:-1]) for r in c if len(r["correct_rank"]) > 2]
            i_p = [min(r["correct_rank"][1:-1]) for r in i if len(r["correct_rank"]) > 2]
            c5 = sum(1 for r in c_p if r <= 4)
            i5 = sum(1 for r in i_p if r <= 4)
            print(f"  {data['model']} (n={len(results)}):")
            print(f"    Correct: peak rank median={np.median(c_p):.0f}, top-5 rate={100*c5/len(c_p):.1f}%")
            print(f"    Incorrect: peak rank median={np.median(i_p):.0f}, top-5 rate={100*i5/len(i_p):.1f}%")


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        return
    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} samples")
    analyze_cis_comp(all_results)


if __name__ == "__main__":
    main()
