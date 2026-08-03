"""Data loader: 下载 TriviaQA, 构造 prompt, 准备答案 token ids."""
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
        aliases: List[str] (所有合法别名)
    """
    print(f"Loading {config.DATASET_NAME} ({config.DATASET_CONFIG}) ...")
    ds = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split="validation")

    samples = []
    n = num_samples or config.NUM_SAMPLES
    for i, item in enumerate(tqdm(ds, desc="Processing", total=min(n, len(ds)))):
        if i >= n:
            break
        question = item["question"].strip()
        answer = item["answer"]["value"].strip()
        aliases = [a.strip() for a in item["answer"]["aliases"] if a.strip()]
        # canonical answer + aliases 去重
        all_answers = list(set([answer] + aliases))
        if not question or not answer:
            continue
        samples.append({
            "id": f"triviaqa_{i:04d}",
            "question": question,
            "answer": answer,
            "aliases": all_answers,
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


def tokenize_answer(tokenizer, answer: str) -> List[int]:
    """把答案转成 token ids (不加 BOS, 前面加一个空格模拟生成)."""
    # 答案前面通常有个空格 (因为 assistant\n 后面直接是答案)
    tokens = tokenizer.encode(" " + answer, add_special_tokens=False)
    return tokens[:config.MAX_ANSWER_TOKENS]


def prepare_samples(samples: List[Dict], tokenizer) -> List[Dict]:
    """为每个样本准备 prompt token ids 和答案 token ids."""
    prepared = []
    for s in tqdm(samples, desc="Tokenizing"):
        prompt_text = build_prompt(s["question"], tokenizer)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(prompt_ids) > config.MAX_PROMPT_LEN:
            prompt_ids = prompt_ids[-config.MAX_PROMPT_LEN:]

        # 每个合法答案都转成 token ids
        answer_token_ids = []
        for ans in s["aliases"]:
            ids = tokenize_answer(tokenizer, ans)
            if len(ids) > 0:
                answer_token_ids.append(ids)

        if not answer_token_ids:
            continue

        prepared.append({
            **s,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "answer_token_ids": answer_token_ids,  # List[List[int]], 每个合法答案一个
            "primary_answer_ids": answer_token_ids[0],  # 用第一个做主答案
        })
    return prepared


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    samples = load_triviaqa(5)
    prepared = prepare_samples(samples, tok)
    for s in prepared:
        print(f"\nQ: {s['question']}")
        print(f"A: {s['answer']}")
        print(f"Aliases: {s['aliases'][:3]}")
        print(f"Primary answer tokens: {s['primary_answer_ids']}")
        print(f"Decoded back: {tok.decode(s['primary_answer_ids'])}")
