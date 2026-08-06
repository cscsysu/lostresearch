"""
跨模型验证: Llama-3.1-8B + Mistral-7B + Qwen3-8B

每个模型只复现关键实验:
1. 200 题轨迹采集 (TriviaQA 100 + HotpotQA 50 + GSM8K 50)
2. 正确 vs 错误样本 peak rank 对照
3. Behavioral necessity (消融 gold direction → correct-to-wrong flips)
4. 跨模型 predictor transfer

用法: python run_cross_model.py --model llama
      python run_cross_model.py --model mistral
      python run_cross_model.py --model qwen
"""
import json
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config


MODELS = {
    "qwen": {
        "path": "/data2/css2025/models/Qwen/Qwen3-8B",
        "name": "Qwen3-8B",
        "enable_thinking": False,
    },
    "qwen4b": {
        "path": "/data2/css2025/models/Qwen/Qwen3-4B",
        "name": "Qwen3-4B",
        "enable_thinking": False,
    },
    "qwen14b": {
        "path": "/data2/css2025/models/Qwen/Qwen3-14B",
        "name": "Qwen3-14B",
        "enable_thinking": False,
    },
    "qwen25_7b": {
        "path": "/data2/css2025/models/Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen2.5-7B-Instruct",
        "enable_thinking": None,  # Qwen2.5 没有 thinking mode
    },
    "llama": {
        "path": "/data2/bowen2023/Model/Llama-3.1-8B-Instruct",
        "name": "Llama-3.1-8B",
        "enable_thinking": None,  # Llama 没有 thinking mode
    },
    "mistral": {
        "path": "/data2/css2025/models/mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral-7B-v0.3",
        "enable_thinking": None,  # Mistral 没有 thinking mode
    },
}


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


def get_final_norm_and_unembed(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm, model.lm_head.weight
    raise ValueError("Cannot find final norm / unembedding")


def build_prompt(question, tokenizer, model_key):
    """构造 prompt, 适配不同模型的 chat template."""
    messages = [{"role": "user", "content": question + " Answer with just the answer."}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if MODELS[model_key]["enable_thinking"] is not None:
        kwargs["enable_thinking"] = MODELS[model_key]["enable_thinking"]
    text = tokenizer.apply_chat_template(messages, **kwargs)
    return text


def is_answer_correct(generated, aliases):
    gen_lower = generated.lower().strip()
    for ans in aliases:
        if ans.lower().strip() in gen_lower:
            return True
    try:
        gen_num = float(gen_lower.replace(",", "").replace("%", ""))
        for ans in aliases:
            ans_num = float(ans.lower().strip().replace(",", "").replace("%", ""))
            if abs(gen_num - ans_num) < 0.01 * max(abs(ans_num), 1):
                return True
    except (ValueError, ZeroDivisionError):
        pass
    return False


def load_data(tokenizer, model_key, n_per_task=50):
    """加载数据, 每个任务 n_per_task 题."""
    from datasets import load_dataset
    samples = []

    # TriviaQA
    try:
        ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation")
        for i, item in enumerate(ds):
            if i >= n_per_task:
                break
            q = item["question"].strip()
            a = item["answer"]["value"].strip()
            aliases = [x.strip() for x in item["answer"]["aliases"] if x.strip()]
            seen = {a.lower()}
            ordered = [a]
            for al in aliases:
                if al.lower() not in seen:
                    seen.add(al.lower())
                    ordered.append(al)
            prompt_text = build_prompt(q, tokenizer, model_key)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(prompt_text + a, add_special_tokens=False)[len(prompt_ids):]
            if answer_ids:
                samples.append({
                    "id": f"triviaqa_{i:04d}", "task": "triviaqa",
                    "question": q, "answer": a, "aliases": ordered,
                    "prompt_text": prompt_text, "prompt_ids": prompt_ids,
                    "primary_answer_ids": answer_ids[:10],
                })
    except Exception as e:
        print(f"TriviaQA failed: {e}")

    # HotpotQA
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        for i, item in enumerate(ds):
            if i >= n_per_task:
                break
            q = item["question"].strip()
            a = item["answer"].strip()
            prompt_text = build_prompt(q, tokenizer, model_key)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(prompt_text + a, add_special_tokens=False)[len(prompt_ids):]
            if answer_ids:
                samples.append({
                    "id": f"hotpotqa_{i:04d}", "task": "hotpotqa",
                    "question": q, "answer": a, "aliases": [a],
                    "prompt_text": prompt_text, "prompt_ids": prompt_ids,
                    "primary_answer_ids": answer_ids[:10],
                })
    except Exception as e:
        print(f"HotpotQA failed: {e}")

    # GSM8K
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, item in enumerate(ds):
            if i >= n_per_task:
                break
            q = item["question"].strip()
            raw_a = item["answer"]
            a = raw_a.split("####")[-1].strip().replace(",", "") if "####" in raw_a else raw_a.strip()
            prompt_text = build_prompt(q, tokenizer, model_key)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(prompt_text + a, add_special_tokens=False)[len(prompt_ids):]
            if answer_ids:
                samples.append({
                    "id": f"gsm8k_{i:04d}", "task": "gsm8k",
                    "question": q, "answer": a, "aliases": [a],
                    "prompt_text": prompt_text, "prompt_ids": prompt_ids,
                    "primary_answer_ids": answer_ids[:10],
                })
    except Exception as e:
        print(f"GSM8K failed: {e}")

    print(f"Loaded {len(samples)} samples")
    return samples


@torch.no_grad()
def collect_trajectory(model, tokenizer, prompt_ids, answer_token_ids, gen_text):
    """采集轨迹 + 生成 + CIS."""
    layers = get_residual_layers(model)
    num_layers = len(layers)
    final_norm, unembed = get_final_norm_and_unembed(model)
    unembed_f = unembed.float()
    device = model.device

    # gen_token
    prompt_text_len = len(prompt_ids)
    if gen_text:
        gen_full = tokenizer.encode(tokenizer.decode(prompt_ids) + gen_text, add_special_tokens=False)
        gen_ids = gen_full[prompt_text_len:]
        gen_token = gen_ids[0] if gen_ids else answer_token_ids[0][0]
    else:
        gen_token = answer_token_ids[0][0]
    target_token = answer_token_ids[0][0]

    # Forward
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
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
    out = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    # CIS per layer
    cis = []
    correct_rank = []
    correct_logprob = []
    for l in range(num_layers):
        h = hidden_buffer[l].to(device).float()
        normed = final_norm(h)
        logits = F.linear(normed, unembed_f)
        log_probs = F.log_softmax(logits, dim=-1)
        lp_correct = log_probs[target_token].item()
        lp_gen = log_probs[gen_token].item()
        cis.append(lp_correct - lp_gen)
        correct_logprob.append(lp_correct)
        sorted_idx = torch.argsort(logits, descending=True)
        rank = (sorted_idx == target_token).nonzero(as_tuple=True)[0].item()
        correct_rank.append(rank)

    return cis, correct_rank, correct_logprob


@torch.no_grad()
def generate_answer(model, tokenizer, prompt_ids):
    device = model.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids, max_new_tokens=32, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def behavioral_necessity(model, tokenizer, correct_samples, model_key):
    """消融 gold direction, 看 correct-to-wrong flips."""
    layers = get_residual_layers(model)
    unembed = model.lm_head.weight
    device = model.device
    num_layers = len(layers)

    # 消融最后 3 层
    ablate_layers = list(range(num_layers-3, num_layers))
    results = []

    for s in tqdm(correct_samples[:30], desc="Behavioral necessity"):
        prompt_ids = s["prompt_ids"]
        target_token = s["primary_answer_ids"][0]
        gold_aliases = s["aliases"]

        # Baseline generate
        baseline_gen = generate_answer(model, tokenizer, prompt_ids)
        baseline_correct = is_answer_correct(baseline_gen, gold_aliases)
        if not baseline_correct:
            continue

        # 消融 gold direction
        answer_dir = unembed[target_token].detach()
        def ablate_hook_factory(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    rest = output[1:]
                else:
                    h = output
                    rest = ()
                last_h = h[0, -1, :]
                proj = (last_h @ answer_dir) / (answer_dir @ answer_dir) * answer_dir
                h[0, -1, :] = last_h - proj
                if rest:
                    return (h,) + rest
                return h
            return hook

        hooks = [layers[l].register_forward_hook(ablate_hook_factory(l)) for l in ablate_layers]
        ablated_gen = generate_answer(model, tokenizer, prompt_ids)
        for h in hooks:
            h.remove()
        ablated_correct = is_answer_correct(ablated_gen, gold_aliases)

        # Random direction 对照
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

        hooks = [layers[l].register_forward_hook(random_hook_factory(l)) for l in ablate_layers]
        random_gen = generate_answer(model, tokenizer, prompt_ids)
        for h in hooks:
            h.remove()
        random_correct = is_answer_correct(random_gen, gold_aliases)

        results.append({
            "id": s["id"],
            "answer": s["answer"],
            "baseline_correct": baseline_correct,
            "ablated_correct": ablated_correct,
            "random_correct": random_correct,
            "flipped": baseline_correct and not ablated_correct,
            "random_flipped": baseline_correct and not random_correct,
        })

    return results


def run_model(model_key):
    """对一个模型跑全部跨模型实验."""
    cfg = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"Cross-Model: {cfg['name']}")
    print(f"{'='*70}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
    except Exception as e:
        print(f"  tokenizer load failed ({e}), trying 8B fallback...")
        tokenizer = AutoTokenizer.from_pretrained(MODELS["qwen"]["path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    # 1. 加载数据
    samples = load_data(tokenizer, model_key, n_per_task=50)

    # 2. 采集轨迹 + 生成
    all_results = []
    for s in tqdm(samples, desc="Collecting"):
        try:
            gen_text = generate_answer(model, tokenizer, s["prompt_ids"])
            correct = is_answer_correct(gen_text, s["aliases"])
            cis, correct_rank, correct_logprob = collect_trajectory(
                model, tokenizer, s["prompt_ids"], [s["primary_answer_ids"]], gen_text)
            all_results.append({
                "id": s["id"], "task": s["task"],
                "question": s["question"], "answer": s["answer"],
                "generated": gen_text, "final_correct": correct,
                "cis": cis, "correct_rank": correct_rank,
                "correct_logprob": correct_logprob,
                "prompt_ids": s["prompt_ids"],
                "primary_answer_ids": s["primary_answer_ids"],
                "aliases": s["aliases"],
                "prompt_text": s["prompt_text"],
            })
        except Exception as e:
            print(f"  Error {s['id']}: {e}")

    correct = [r for r in all_results if r["final_correct"]]
    incorrect = [r for r in all_results if not r["final_correct"]]
    print(f"\nTotal: {len(all_results)}, Correct: {len(correct)}, Incorrect: {len(incorrect)}")

    # 3. 正确 vs 错误对照
    print(f"\n--- {cfg['name']}: Correct vs Incorrect ---")
    for label, group in [("Correct", correct), ("Incorrect", incorrect)]:
        if not group:
            continue
        peak_ranks = [min(r["correct_rank"][1:-1]) for r in group if len(r["correct_rank"]) > 2]
        dwell_times = [sum(1 for c in r["cis"] if c > 0) / len(r["cis"]) for r in group if len(r["cis"]) > 2]
        decays = [max(r["cis"][1:-1]) - r["cis"][-1] for r in group if len(r["cis"]) > 2]
        sign_changes = sum(1 for r in group
                           if any(r["cis"][i-1] > 0 and r["cis"][i] < 0
                                  for i in range(1, len(r["cis"]))))
        print(f"  {label} (n={len(group)}):")
        print(f"    peak rank: median={np.median(peak_ranks):.0f}, mean={np.mean(peak_ranks):.0f}")
        print(f"    dwell time: mean={np.mean(dwell_times):.3f}")
        print(f"    peak-to-final decay: mean={np.mean(decays):.3f}")
        print(f"    sign change: {sign_changes}/{len(group)} ({100*sign_changes/len(group):.1f}%)")

    # 4. Behavioral necessity
    print(f"\n--- {cfg['name']}: Behavioral Necessity ---")
    beh_results = behavioral_necessity(model, tokenizer, correct, model_key)
    baseline = sum(1 for r in beh_results if r["baseline_correct"])
    ablated = sum(1 for r in beh_results if r["ablated_correct"])
    random_abl = sum(1 for r in beh_results if r["random_correct"])
    flipped = sum(1 for r in beh_results if r["flipped"])
    random_flipped = sum(1 for r in beh_results if r["random_flipped"])
    print(f"  Baseline correct: {baseline}/{len(beh_results)}")
    print(f"  Ablated correct:   {ablated}/{len(beh_results)} ({flipped} flipped)")
    print(f"  Random ablation:  {random_abl}/{len(beh_results)} ({random_flipped} flipped)")
    if flipped > random_flipped:
        print(f"  ✓ Gold direction 消融导致更多 flips ({flipped} > {random_flipped})")
    else:
        print(f"  ? 未显著")

    # 5. 保存
    model_short = model_key
    out_file = os.path.join(config.DATA_DIR, f"cross_model_{model_short}.json")
    with open(out_file, "w") as f:
        json.dump({
            "model": cfg["name"],
            "trajectory_results": all_results,
            "behavioral_necessity": beh_results,
        }, f, ensure_ascii=False, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"\nSaved: {out_file}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()))
    args = parser.parse_args()
    run_model(args.model)
