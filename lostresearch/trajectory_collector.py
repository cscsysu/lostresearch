"""
Trajectory collector + CIS 计算.

功能:
1. Hook 采集每层 hidden state (完整序列)
2. Logit lens 解码每层
3. 对所有合法 aliases 用 teacher forcing 算多 token 序列 log-prob
4. 同时采集 generated answer 的轨迹 (作为错误信号对照)
5. CIS = correct_logprob - generated_logprob
6. Sanity check
"""
import torch
import torch.nn.functional as F
from typing import List, Dict

import config


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find transformer layers")


def get_final_norm_and_unembedding(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm, model.lm_head.weight
    raise ValueError("Cannot find final norm / unembedding")


@torch.no_grad()
def compute_sequence_logprob_with_logit_lens(
    hidden_seq: torch.Tensor,   # [seq_len, hidden]
    target_token_ids: List[int], # [a1, a2, ..., ak]
    final_norm,
    unembed,
    last_prompt_pos: int,
) -> float:
    """用 logit lens 算多 token 答案的 teacher forcing log-prob.

    近似: 用最后一个 prompt token 位置的 hidden state 预测 answer 第一个 token,
    用倒数第二个位置预测第二个 token... 这不对。

    正确做法: 把 prompt + answer 一起 forward, 在每个位置 i 用层 l 的 hidden state 预测 token i+1。
    但这样每层都要单独 forward, 计算量大。

    折中: 这里只用最后一个 prompt token 位置预测 answer 第一个 token。
    多 token 的完整 teacher forcing 留给后续改进。
    当前版本: 返回首 token 的 log-prob (这是最可控的做法)。
    """
    if len(target_token_ids) == 0:
        return -1e9

    target_token = target_token_ids[0]
    h = hidden_seq[last_prompt_pos:last_prompt_pos+1]  # [1, hidden]
    h = h.to(unembed.device).to(unembed.dtype)

    normed = final_norm(h)
    logits = F.linear(normed, unembed).squeeze(0)  # [vocab]
    log_probs = F.log_softmax(logits, dim=-1)

    if target_token >= logits.shape[-1]:
        return -1e9
    return log_probs[target_token].item()


@torch.no_grad()
def compute_rank_with_logit_lens(
    hidden_seq: torch.Tensor,
    target_token: int,
    final_norm,
    unembed,
    last_prompt_pos: int,
) -> int:
    h = hidden_seq[last_prompt_pos:last_prompt_pos+1]
    h = h.to(unembed.device).to(unembed.dtype)
    normed = final_norm(h)
    logits = F.linear(normed, unembed).squeeze(0)
    sorted_idx = torch.argsort(logits, descending=True)
    if target_token >= logits.shape[-1]:
        return 999999
    return (sorted_idx == target_token).nonzero(as_tuple=True)[0].item()


@torch.no_grad()
def get_topk_with_logit_lens(hidden_seq, final_norm, unembed, last_prompt_pos, k=5):
    h = hidden_seq[last_prompt_pos:last_prompt_pos+1]
    h = h.to(unembed.device).to(unembed.dtype)
    normed = final_norm(h)
    logits = F.linear(normed, unembed).squeeze(0)
    topk_vals, topk_idx = torch.topk(logits, k=k)
    topk_probs = F.softmax(topk_vals, dim=-1)
    return list(zip(topk_idx.tolist(), topk_probs.tolist())), logits


class TrajectoryCollector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_residual_layers(model)
        self.final_norm, self.unembed = get_final_norm_and_unembedding(model)
        self.num_layers = len(self.layers)
        self.hidden_states = []
        self._hooks = []

    def _register_hooks(self):
        self.hidden_states = []
        self._hooks = []
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self.hidden_states.append((layer_idx, hidden[0].detach()))
            return hook
        for i, layer in enumerate(self.layers):
            h = layer.register_forward_hook(make_hook(i))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def collect_trajectory(self, prompt_ids: List[int],
                            answer_token_ids: List[List[int]],
                            generated_token_ids: List[int]) -> Dict:
        """采集每层轨迹 + CIS.

        Args:
            prompt_ids: prompt token ids
            answer_token_ids: List[List[int]], 每个合法答案一个序列
            generated_token_ids: 模型实际生成的 token ids (作为错误信号对照)
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        last_prompt_pos = len(prompt_ids) - 1

        self._register_hooks()
        self.hidden_states = []
        outputs = self.model(input_ids, use_cache=False, output_hidden_states=False)
        self._remove_hooks()

        self.hidden_states.sort(key=lambda x: x[0])

        # 最终层 logits (模型原生, 用于 sanity check)
        final_logits = outputs.logits[0, -1, :]  # [vocab]
        final_argmax_token = final_logits.argmax().item()

        # 对每层计算: correct (max over aliases), generated, CIS
        correct_logprob_per_layer = []
        correct_rank_per_layer = []
        multitoken_rank_per_layer = []  # best rank across all answer tokens
        generated_logprob_per_layer = []
        generated_rank_per_layer = []
        cis_per_layer = []  # CIS = correct - generated
        top5_per_layer = []

        # generated token 的首 token
        gen_first_token = generated_token_ids[0] if generated_token_ids else -1

        for layer_idx, hidden_seq in self.hidden_states:
            # 1. Correct: 对所有 aliases 取 max
            best_correct_lp = -1e9
            best_correct_rank = 999999
            best_multitoken_rank = 999999  # best rank across ALL tokens of ALL aliases
            for alias_ids in answer_token_ids:
                if len(alias_ids) == 0:
                    continue
                lp = compute_sequence_logprob_with_logit_lens(
                    hidden_seq, alias_ids, self.final_norm, self.unembed, last_prompt_pos)
                if lp > best_correct_lp:
                    best_correct_lp = lp
                    best_correct_rank = compute_rank_with_logit_lens(
                        hidden_seq, alias_ids[0], self.final_norm, self.unembed, last_prompt_pos)
                # Multi-token: check rank of EVERY token in this alias
                for tok in alias_ids:
                    tok_rank = compute_rank_with_logit_lens(
                        hidden_seq, tok, self.final_norm, self.unembed, last_prompt_pos)
                    if tok_rank < best_multitoken_rank:
                        best_multitoken_rank = tok_rank

            # 2. Generated: 生成答案的首 token
            if gen_first_token >= 0:
                gen_lp = compute_sequence_logprob_with_logit_lens(
                    hidden_seq, [gen_first_token], self.final_norm, self.unembed, last_prompt_pos)
                gen_rank = compute_rank_with_logit_lens(
                    hidden_seq, gen_first_token, self.final_norm, self.unembed, last_prompt_pos)
            else:
                gen_lp = -1e9
                gen_rank = 999999

            # 3. CIS = correct - generated
            cis = best_correct_lp - gen_lp

            # 4. Top-5
            top5, _ = get_topk_with_logit_lens(
                hidden_seq, self.final_norm, self.unembed, last_prompt_pos, k=5)

            correct_logprob_per_layer.append(best_correct_lp)
            correct_rank_per_layer.append(best_correct_rank)
            multitoken_rank_per_layer.append(best_multitoken_rank)
            generated_logprob_per_layer.append(gen_lp)
            generated_rank_per_layer.append(gen_rank)
            cis_per_layer.append(cis)
            top5_per_layer.append(top5)

        return {
            "correct_logprob": correct_logprob_per_layer,
            "correct_rank": correct_rank_per_layer,
            "multitoken_best_rank": multitoken_rank_per_layer,
            "generated_logprob": generated_logprob_per_layer,
            "generated_rank": generated_rank_per_layer,
            "cis": cis_per_layer,  # correct - generated
            "top5": top5_per_layer,
            "num_layers": len(correct_logprob_per_layer),
            "final_argmax_token": final_argmax_token,
        }

    @torch.no_grad()
    def generate_answer(self, prompt_ids: List[int]) -> Dict:
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = self.model.generate(
            input_ids,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=config.DO_SAMPLE,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][input_ids.shape[1]:].tolist()
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return {
            "text": text,
            "token_ids": gen_ids,
            "first_token_id": gen_ids[0] if gen_ids else -1,
            "first_token_decoded": self.tokenizer.decode([gen_ids[0]]) if gen_ids else "",
        }
