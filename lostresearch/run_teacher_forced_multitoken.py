"""
Teacher-forced multi-token trajectory (sequence-level definition).

Reviewer's P0 requirement: "对 gold answer 的每个 token，在 teacher-forced gold
prefix 下计算该位置的层间 trajectory... 不要使用'每层取任一 token 最优值'"

This implements the proper sequence-level protocol:
1. Build the input as [prompt] + [gold_answer_tokens] (teacher forcing).
2. For answer token i at position p_i, read the hidden state at position p_i - 1
   (the position that PREDICTS token i) at every layer.
3. Decode that hidden state and get rank/logprob of gold token i.
4. Aggregate across answer positions with THREE well-defined variants:
   - ALL:  every answer token must be competitive at the same layer
   - MEAN: average per-position rank/logprob across answer tokens
   - JOINT: sum of per-position log-probabilities (sequence log-likelihood)

This removes the cherry-picking concern of "min over tokens", because each
aggregation rule is applied consistently at every layer.

Usage:
  python run_teacher_forced_multitoken.py --n 300
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_all_datasets, prepare_samples


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--k", type=int, default=config.RANK_COMPETITIVE)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 70)
    print("Teacher-Forced Multi-Token Trajectory (sequence-level)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE)
    model.eval()

    layers = get_residual_layers(model)
    num_layers = len(layers)
    final_norm = model.model.norm
    unembed = model.lm_head.weight.float()
    device = model.device

    # Load existing results for correctness labels
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    existing = {}
    if os.path.exists(results_file):
        with open(results_file) as fh:
            for r in json.load(fh):
                existing[r["id"]] = r

    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)
    for s in prepared:
        if s["id"] in existing:
            s["final_correct"] = existing[s["id"]]["final_correct"]

    errors = [s for s in prepared if not s.get("final_correct", True)]
    print(f"\nErrors available: {len(errors)}; evaluating up to {args.n}")

    @torch.no_grad()
    def teacher_forced_trajectory(sample):
        """Return per-layer, per-answer-position rank and logprob."""
        prompt_ids = sample["prompt_ids"]
        ans_ids = sample["primary_answer_ids"]
        m = len(ans_ids)
        if m == 0:
            return None

        # Teacher forcing: prompt + gold answer
        full_ids = list(prompt_ids) + list(ans_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        prompt_len = len(prompt_ids)

        # Capture hidden states at every layer
        captured = {}

        def make_hook(idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h[0].detach()  # [seq_len, hidden]
            return hook

        handles = [layers[l].register_forward_hook(make_hook(l))
                   for l in range(num_layers)]
        model(input_ids, use_cache=False)
        for h in handles:
            h.remove()

        # For answer token i (0-indexed), the predicting position is
        # prompt_len - 1 + i
        ranks = np.zeros((num_layers, m), dtype=np.int64)
        lps = np.zeros((num_layers, m), dtype=np.float64)
        for l in range(num_layers):
            hseq = captured[l].to(device).float()
            for i, tok in enumerate(ans_ids):
                pos = prompt_len - 1 + i
                if pos >= hseq.shape[0]:
                    ranks[l, i] = 10**9
                    lps[l, i] = -1e9
                    continue
                h = final_norm(hseq[pos:pos + 1])
                logits = F.linear(h, unembed).squeeze(0)
                logprobs = F.log_softmax(logits, dim=-1)
                ranks[l, i] = int((logits > logits[tok]).sum().item())
                lps[l, i] = float(logprobs[tok].item())
        return ranks, lps

    # Aggregate with three well-defined rules
    stats = {
        "ALL": {"pres": 0, "n": 0},
        "MEAN": {"pres": 0, "n": 0},
        "JOINT": {"pres": 0, "n": 0},
        "MIN(old)": {"pres": 0, "n": 0},
    }
    per_task = {}
    records = []

    for s in tqdm(errors[:args.n], desc="Teacher-forced"):
        res = teacher_forced_trajectory(s)
        if res is None:
            continue
        ranks, lps = res
        m = ranks.shape[1]
        task = s.get("task", "unknown")
        per_task.setdefault(task, {k: {"pres": 0, "n": 0} for k in stats})

        # ALL: at some intermediate layer, EVERY answer token has rank <= k
        all_ok = any(all(ranks[l, i] <= args.k for i in range(m))
                     for l in range(num_layers - 1))
        # MEAN: at some intermediate layer, MEAN rank <= k
        mean_ok = any(ranks[l, :].mean() <= args.k for l in range(num_layers - 1))
        # JOINT: at some intermediate layer, joint sequence logprob exceeds
        #        the final-layer joint logprob (i.e., it was better mid-way)
        joint = lps.sum(axis=1)
        joint_ok = bool(joint[:-1].max() > joint[-1])
        # MIN (the old rule, for comparison)
        min_ok = any(ranks[l, :].min() <= args.k for l in range(num_layers - 1))

        for key, flag in [("ALL", all_ok), ("MEAN", mean_ok),
                          ("JOINT", joint_ok), ("MIN(old)", min_ok)]:
            stats[key]["n"] += 1
            stats[key]["pres"] += int(flag)
            per_task[task][key]["n"] += 1
            per_task[task][key]["pres"] += int(flag)

        records.append({
            "id": s["id"], "task": task, "n_answer_tokens": m,
            "all_ok": bool(all_ok), "mean_ok": bool(mean_ok),
            "joint_ok": bool(joint_ok), "min_ok": bool(min_ok),
            "final_joint_lp": float(joint[-1]),
            "peak_joint_lp": float(joint[:-1].max()),
        })

    # Report
    print("\n" + "=" * 70)
    print(f"Sequence-level preservation rates (k={args.k})")
    print("=" * 70)
    for key in ["ALL", "MEAN", "JOINT", "MIN(old)"]:
        st = stats[key]
        if st["n"]:
            print(f"  {key:10s}: {st['pres']}/{st['n']} ({100*st['pres']/st['n']:.1f}%)")

    print("\n  Per-task (aggregation = ALL / MEAN / JOINT):")
    for task in sorted(per_task):
        pt = per_task[task]
        n = pt["ALL"]["n"]
        if n == 0:
            continue
        print(f"    {task:16s} n={n:4d}  "
              f"ALL={100*pt['ALL']['pres']/n:5.1f}%  "
              f"MEAN={100*pt['MEAN']['pres']/n:5.1f}%  "
              f"JOINT={100*pt['JOINT']['pres']/n:5.1f}%")

    # Answer-length breakdown
    print("\n  By answer length (number of tokens):")
    for lo, hi, label in [(1, 1, "1 token"), (2, 2, "2 tokens"), (3, 99, "3+ tokens")]:
        sub = [r for r in records if lo <= r["n_answer_tokens"] <= hi]
        if sub:
            print(f"    {label:10s} n={len(sub):4d}  "
                  f"ALL={100*np.mean([r['all_ok'] for r in sub]):5.1f}%  "
                  f"MEAN={100*np.mean([r['mean_ok'] for r in sub]):5.1f}%  "
                  f"JOINT={100*np.mean([r['joint_ok'] for r in sub]):5.1f}%")

    out_file = os.path.join(config.DATA_DIR, "teacher_forced_multitoken_Qwen3-8B.json")
    with open(out_file, "w") as fh:
        json.dump({"k": args.k, "stats": stats, "per_task": per_task,
                   "records": records}, fh, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
