"""Data loader: 多数据集加载, 精确 token 对齐."""
import re
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm

import config


def normalize_answer(s: str) -> str:
    """标准化答案: 去除多余空格、标点、大小写."""
    s = s.strip()
    # 去除常见前缀
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)
    return s


def load_triviaqa(n: int) -> List[Dict]:
    ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation")
    samples = []
    for i, item in enumerate(ds):
        if i >= n: break
        question = item["question"].strip()
        answer = item["answer"]["value"].strip()
        aliases = [a.strip() for a in item["answer"]["aliases"] if a.strip()]
        # canonical 在第一位, 去重
        seen = {answer.lower()}
        ordered = [answer]
        for a in aliases:
            if a.lower() not in seen:
                seen.add(a.lower())
                ordered.append(a)
        samples.append({
            "id": f"triviaqa_{i:04d}",
            "question": question,
            "answer": answer,
            "aliases": ordered,
            "task": "triviaqa",
        })
    return samples


def load_hotpotqa(n: int) -> List[Dict]:
    # hotpotqa 官方 namespace 是 hotpotqa
    for name in ["hotpotqa/hotpot_qa", "hotpot_qa"]:
        try:
            ds = load_dataset(name, "distractor", split="validation")
            break
        except Exception:
            try:
                ds = load_dataset(name, "fullwiki", split="validation")
                break
            except Exception:
                continue
    else:
        raise RuntimeError("Cannot load hotpot_qa")
    samples = []
    for i, item in enumerate(ds):
        if i >= n: break
        question = item["question"].strip()
        answer = item["answer"].strip()
        # HotpotQA 通常没有 aliases
        aliases = [answer]
        samples.append({
            "id": f"hotpotqa_{i:04d}",
            "question": question,
            "answer": answer,
            "aliases": aliases,
            "task": "hotpotqa",
        })
    return samples


def load_gsm8k(n: int) -> List[Dict]:
    # gsm8k 官方 namespace 是 openai
    for name in ["openai/gsm8k", "gsm8k"]:
        try:
            ds = load_dataset(name, "main", split="test")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Cannot load gsm8k")
    samples = []
    for i, item in enumerate(ds):
        if i >= n: break
        question = item["question"].strip()
        # GSM8K answer 格式: "...\n#### 42"
        raw_answer = item["answer"]
        if "####" in raw_answer:
            answer = raw_answer.split("####")[-1].strip()
        else:
            answer = raw_answer.strip()
        # 去除逗号
        answer = answer.replace(",", "")
        aliases = [answer]
        samples.append({
            "id": f"gsm8k_{i:04d}",
            "question": question,
            "answer": answer,
            "aliases": aliases,
            "task": "gsm8k",
        })
    return samples


def load_all_datasets() -> List[Dict]:
    """加载所有数据集."""
    all_samples = []
    for ds_cfg in config.DATASETS:
        label = ds_cfg["label"]
        n = ds_cfg["n"]
        print(f"Loading {label} (n={n}) ...")
        try:
            if label == "triviaqa":
                samples = load_triviaqa(n)
            elif label == "hotpotqa":
                samples = load_hotpotqa(n)
            elif label == "gsm8k":
                samples = load_gsm8k(n)
            else:
                continue
            all_samples.extend(samples)
            print(f"  Loaded {len(samples)} samples")
        except Exception as e:
            print(f"  Failed to load {label}: {e}")
    print(f"Total: {len(all_samples)} samples")
    return all_samples


def build_prompt(question: str, tokenizer) -> str:
    messages = [{"role": "user", "content": question + " Answer with just the answer."}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING,
    )
    return text


def compute_answer_token_ids(tokenizer, prompt_text: str, answer: str) -> List[int]:
    """精确对齐: encode(prompt+answer) - encode(prompt)."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + answer, add_special_tokens=False)
    answer_ids = full_ids[len(prompt_ids):]
    return answer_ids[:config.MAX_ANSWER_TOKENS]


def prepare_samples(samples: List[Dict], tokenizer) -> List[Dict]:
    prepared = []
    skipped = 0
    for s in tqdm(samples, desc="Tokenizing"):
        prompt_text = build_prompt(s["question"], tokenizer)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > config.MAX_PROMPT_LEN:
            prompt_ids = prompt_ids[-config.MAX_PROMPT_LEN:]
        answer_token_ids = []
        for ans in s["aliases"]:
            ids = compute_answer_token_ids(tokenizer, prompt_text, ans)
            if len(ids) > 0:
                answer_token_ids.append(ids)
        if not answer_token_ids:
            skipped += 1
            continue
        prepared.append({
            **s,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "answer_token_ids": answer_token_ids,
            "primary_answer_ids": answer_token_ids[0],
        })
    print(f"Prepared {len(prepared)} samples (skipped {skipped})")
    return prepared
