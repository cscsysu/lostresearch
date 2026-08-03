"""Data loader: 下载 TriviaQA, 构造 prompt, 精确对齐答案 continuation tokens."""
import json
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm

import config


def load_triviaqa(num_samples: int = None) -> List[Dict]:
    """加载 TriviaQA, 返回样本列表.

    每个样本:
        question: str
        answer: str (canonical answer)
        aliases: List[str] (所有合法别名, canonical 在第一位)
    """
    # HuggingFace 新版 datasets 要求 namespace/name 格式
    dataset_name = config.DATASET_NAME
    if "/" not in dataset_name:
        dataset_name = f"mandarjoshi/{dataset_name}"

    # 尝试多个配置名以兼容不同版本的 datasets
    configs_to_try = [config.DATASET_CONFIG, "rc.nocontext", "rc", "unfiltered"]
    ds = None
    last_err = None
    for cfg in configs_to_try:
        try:
            print(f"Loading {dataset_name} (config={cfg}) ...")
            ds = load_dataset(dataset_name, cfg, split="validation")
            print(f"  ✓ 成功 (config={cfg})")
            break
        except Exception as e:
            print(f"  ✗ config={cfg} 失败: {str(e)[:100]}")
            last_err = e
    if ds is None:
        raise last_err

    samples = []
    n = num_samples or config.NUM_SAMPLES
    for i, item in enumerate(tqdm(ds, desc="Processing", total=min(n, len(ds)))):
        if i >= n:
            break
        question = item["question"].strip()
        answer = item["answer"]["value"].strip()
        aliases = [a.strip() for a in item["answer"]["aliases"] if a.strip()]
        # canonical answer 必须在第一位, 其余 alias 保持稳定顺序去重
        seen = {answer.lower()}
        ordered = [answer]
        for a in aliases:
            if a.lower() not in seen:
                seen.add(a.lower())
                ordered.append(a)
        if not question or not answer:
            continue
        samples.append({
            "id": f"triviaqa_{i:04d}",
            "question": question,
            "answer": answer,
            "aliases": ordered,
        })
    print(f"Loaded {len(samples)} samples")
    return samples


def build_prompt(question: str, tokenizer) -> str:
    """构造 non-thinking mode 的 chat prompt."""
    messages = [{"role": "user", "content": question + " Answer with just the answer."}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING,
    )
    return text


def compute_answer_token_ids(tokenizer, prompt_text: str, answer: str) -> List[int]:
    """精确对齐: 把 prompt + answer 一起 encode, 取多出来的部分作为 answer_ids.

    这样得到的 answer_ids 是模型在当前 prompt 后真正会生成的 token 序列,
    而不是人为假设 ' answer' 这种带空格的 token.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_text + answer, add_special_tokens=False)

    # answer_ids 就是 full_ids 比 prompt_ids 多出来的部分
    answer_ids = full_ids[len(prompt_ids):]

    # 截断到最大长度
    return answer_ids[:config.MAX_ANSWER_TOKENS]


def prepare_samples(samples: List[Dict], tokenizer) -> List[Dict]:
    """为每个样本准备 prompt token ids 和所有合法答案的 continuation token ids."""
    prepared = []
    skipped = 0
    for s in tqdm(samples, desc="Tokenizing"):
        prompt_text = build_prompt(s["question"], tokenizer)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(prompt_ids) > config.MAX_PROMPT_LEN:
            prompt_ids = prompt_ids[-config.MAX_PROMPT_LEN:]

        # 对每个合法答案都计算精确 continuation token ids
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
            "answer_token_ids": answer_token_ids,  # List[List[int]], 第一个是 canonical
            "primary_answer_ids": answer_token_ids[0],
        })
    print(f"Prepared {len(prepared)} samples (skipped {skipped} due to empty token ids)")
    return prepared
