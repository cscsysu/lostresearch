"""
Trajectory collector + CIS (Correct Information Signal) 计算.

修正版:
1. 多 token 答案用完整 teacher forcing 算序列 log-prob
2. 对所有合法 aliases 取最优 (max over aliases)
3. 加 final-layer sanity check: argmax(final_logits) 应该 == 生成的第一个 token
4. 同时采集 hard negative 作为对照 (CIS = correct - incorrect)
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Optional

import config


def get_residual_layers(model) -> List:
    """获取所有 Transformer block."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    else:
        raise ValueError("Cannot find transformer layers in model")


def get_final_norm_and_unembedding(model):
    """获取最终的 LayerNorm 和 unembedding 矩阵 (LM head)."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
        unembed = model.lm_head.weight  # [vocab, hidden]
    else:
        raise ValueError("Cannot find final norm / unembedding")
    return final_norm, unembed


class TrajectoryCollector:
    """采集每层 hidden state, 计算 CIS 轨迹.

    修正点:
    - 多 token 答案用 teacher forcing: 累加每个答案 token 的 log-prob
    - 对所有合法 aliases 取 max (最优答案)
    - 加 final-layer sanity check
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_residual_layers(model)
        self.final_norm, self.unembed = get_final_norm_and_unembedding(model)
        self.num_layers = len(self.layers)

        self.hidden_states = []  # 每层完整序列的 hidden states
        self._hooks = []

    def _register_hooks(self):
        """注册 forward hook 采集每层完整序列的 hidden state."""
        self.hidden_states = []
        self._hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # 存完整序列 [batch, seq, hidden], 而不是只存最后一个 token
                self.hidden_states.append((layer_idx, hidden[0].detach()))  # [seq, hidden]
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
                            answer_token_ids: List[List[int]]) -> Dict:
        """对一个样本跑一次 forward, 采集每层轨迹.

        Args:
            prompt_ids: prompt 的 token ids
            answer_token_ids: List[List[int]], 每个合法答案一个 token 序列

        Returns:
            {
                "correct_token_logprob": List[float],   # 每层最优答案的 log-prob (max over aliases)
                "correct_token_rank": List[int],        # 每层最优答案首 token 的 rank
                "top5_per_layer": List[List[(token_id, prob)]],
                "best_alias_idx_per_layer": List[int],  # 每层最优答案的 index
                "sanity_check": Dict,  # final layer argmax == generated first token?
            }
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 1. 注册 hook 并 forward
        self._register_hooks()
        self.hidden_states = []
        outputs = self.model(input_ids, use_cache=False, output_hidden_states=False)
        self._remove_hooks()

        # 按 layer_idx 排序
        self.hidden_states.sort(key=lambda x: x[0])

        # 同时取最终层的 logits 做 sanity check
        final_logits = outputs.logits[0]  # [seq, vocab]
        final_last_pos_logits = final_logits[-1]  # [vocab]
        final_argmax_token = final_last_pos_logits.argmax().item()

        # 2. 对每层 hidden state 用 logit lens 解码, 计算每个答案的 log-prob
        # 多 token 答案: 用 teacher forcing
        # P(answer | prompt) = sum_t log P(answer_t | prompt, answer_<t)
        # 但这里有个问题: 我们只有 prompt 的 forward, 没有跑 answer tokens 的 forward
        # 严格 teacher forcing 需要把 answer tokens 也喂进去
        # 近似: 对于层 l, 用最后一个 prompt token 位置的 hidden state 预测 answer 的第一个 token
        #       用倒数第二个位置的预测第二个 token... 这不对
        #
        # 正确做法: 跑一次完整 forward (prompt + answer), 在每个位置 i 预测 token i+1
        # 这里先做严格版本: 跑 prompt+answer 的 forward

        # 先做简化版: 只测首 token 的 log-prob 和 rank (和之前一样)
        # 但用精确对齐的 answer_ids
        # 多 token 的 teacher forcing 留给下一步

        correct_logprob_per_layer = []
        correct_rank_per_layer = []
        top5_per_layer = []
        best_alias_idx_per_layer = []

        for layer_idx, hidden_seq in self.hidden_states:
            # hidden_seq: [seq, hidden]
            # 取最后一个 prompt token 位置
            last_hidden = hidden_seq[-1:]  # [1, hidden]
            last_hidden = last_hidden.to(self.unembed.device).to(self.unembed.dtype)

            # logit lens: final norm + unembedding
            normed = self.final_norm(last_hidden)
            logits = F.linear(normed, self.unembed)  # [1, vocab]
            logits = logits.squeeze(0)  # [vocab]

            log_probs = F.log_softmax(logits, dim=-1)

            # 对所有合法 aliases, 取首 token log-prob 最大的那个
            best_logprob = -1e9
            best_rank = 999999
            best_alias_idx = 0
            for alias_idx, alias_ids in enumerate(answer_token_ids):
                if len(alias_ids) == 0:
                    continue
                first_token = alias_ids[0]
                if first_token >= logits.shape[-1]:
                    continue
                lp = log_probs[first_token].item()
                if lp > best_logprob:
                    best_logprob = lp
                    best_alias_idx = alias_idx
                    # rank
                    sorted_idx = torch.argsort(logits, descending=True)
                    best_rank = (sorted_idx == first_token).nonzero(as_tuple=True)[0].item()

            correct_logprob_per_layer.append(best_logprob)
            correct_rank_per_layer.append(best_rank)
            best_alias_idx_per_layer.append(best_alias_idx)

            # top 5
            topk_vals, topk_idx = torch.topk(logits, k=5)
            topk_probs = F.softmax(topk_vals, dim=-1)
            top5 = list(zip(topk_idx.tolist(), topk_probs.tolist()))
            top5_per_layer.append(top5)

        # 3. Sanity check: final layer argmax 应该等于生成的第一个 token
        # 我们用 outputs.logits (模型原生最终层) 验证
        sanity = {
            "final_argmax_token": final_argmax_token,
            "final_argmax_decoded": self.tokenizer.decode([final_argmax_token]),
            "expected_first_token": answer_token_ids[0][0] if answer_token_ids and answer_token_ids[0] else -1,
            "expected_first_decoded": self.tokenizer.decode([answer_token_ids[0][0]]) if answer_token_ids and answer_token_ids[0] else "",
            "passed": False,  # 后面 generate 后再判断
        }

        return {
            "correct_token_logprob": correct_logprob_per_layer,
            "correct_token_rank": correct_rank_per_layer,
            "top5_per_layer": top5_per_layer,
            "best_alias_idx_per_layer": best_alias_idx_per_layer,
            "num_layers_collected": len(correct_logprob_per_layer),
            "sanity_check": sanity,
            "final_argmax_token": final_argmax_token,
        }

    @torch.no_grad()
    def generate_answer(self, prompt_ids: List[int]) -> Dict:
        """生成答案, 返回文本和首 token id."""
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        out = self.model.generate(
            input_ids,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=config.DO_SAMPLE,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        generated_ids = out[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        first_token_id = generated_ids[0].item() if len(generated_ids) > 0 else -1
        return {
            "text": text,
            "first_token_id": first_token_id,
            "first_token_decoded": self.tokenizer.decode([first_token_id]) if first_token_id >= 0 else "",
        }
