"""
Trajectory collector + CIS (Correct Information Signal) 计算.

核心逻辑:
1. 注册 forward hook 采集每层 hidden state
2. 对每层 hidden state 用 logit lens (经过最终 norm + unembedding) 解码
3. 计算"正确答案"的多 token 序列概率 (teacher forcing)
4. 同时计算"错误答案"的概率作为对照
5. 返回每样本每层的 CIS 轨迹
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Optional

import config


def get_residual_layers(model) -> List:
    """获取所有 Transformer block, 返回它们的列表 (用于注册 hook)."""
    # Qwen3 用 model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    else:
        raise ValueError("Cannot find transformer layers in model")


def get_final_norm_and_unembedding(model):
    """获取最终的 LayerNorm 和 unembedding 矩阵 (LM head).

    对于 Qwen3:
        final_norm = model.model.norm
        unembed = model.lm_head.weight  # shape: [vocab, hidden]
    """
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
        unembed = model.lm_head.weight  # [vocab, hidden]
    else:
        raise ValueError("Cannot find final norm / unembedding")
    return final_norm, unembed


def logit_lens_decode(hidden_state: torch.Tensor,
                      final_norm: torch.nn.Module,
                      unembed: torch.Tensor) -> torch.Tensor:
    """对某一层的 hidden state 用 logit lens 解码成 logits.

    Args:
        hidden_state: [batch, seq, hidden]
        final_norm: 最终的 LayerNorm
        unembed: [vocab, hidden]

    Returns:
        logits: [batch, seq, vocab]
    """
    # 应用最终 norm
    normed = final_norm(hidden_state)
    # 投影到词表
    logits = F.linear(normed, unembed)  # [batch, seq, vocab]
    return logits


def compute_token_prob_under_logits(logits: torch.Tensor,
                                     token_ids: List[int]) -> float:
    """给定某位置某层的 logits, 计算 teacher forcing 下答案序列的概率.

    logits: [seq, vocab]  (对当前样本, 某一层的输出)
    token_ids: [a1, a2, ..., ak]  答案的 token 序列

    返回 sum(log P(a_i | a_<i)) 的 log-probability.

    这里我们简化: 对每个 token, 取它在当前 logits 下 softmax 后的概率.
    (严格 teacher forcing 应该重新 forward, 这里用近似:
     假设答案位置的 hidden state 反映了"已经看到前面答案 token"的状态.
     对于 single-pass 采集, 这是近似但合理的.)

    更严谨的做法是重新跑 forward, 但那样计算量大 L 倍.
    Pilot 阶段先用这个近似.
    """
    if len(token_ids) == 0:
        return -1e9

    # 我们用第一个答案 token 在最后一个 prompt 位置的概率
    # (这是最简单也最可控的做法, 因为最后一个 prompt 位置是 "assistant\n 后的位置")
    # 多 token 答案的完整序列概率留给第二阶段
    target_token = token_ids[0]
    if target_token >= logits.shape[-1]:
        return -1e9

    # log softmax
    log_probs = F.log_softmax(logits, dim=-1)  # [seq, vocab]
    # 最后一个 prompt token 位置
    last_pos_logprob = log_probs[-1, target_token].item()
    return last_pos_logprob


def compute_token_rank(logits: torch.Tensor, token_id: int) -> int:
    """计算 token 在 logits 下的 rank (0-indexed)."""
    last_pos_logits = logits[-1, :]  # [vocab]
    sorted_indices = torch.argsort(last_pos_logits, descending=True)
    rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item()
    return rank


def compute_topk(logits: torch.Tensor, k: int = 5):
    """返回最后一个位置的 top-k token 和概率."""
    last_pos_logits = logits[-1, :]  # [vocab]
    topk_vals, topk_idx = torch.topk(last_pos_logits, k=k)
    probs = F.softmax(topk_vals, dim=-1)
    return list(zip(topk_idx.tolist(), probs.tolist()))


class TrajectoryCollector:
    """采集每层 hidden state, 计算 CIS 轨迹."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_residual_layers(model)
        self.final_norm, self.unembed = get_final_norm_and_unembedding(model)
        self.num_layers = len(self.layers)

        self.hidden_states = []  # 每层最后一个 token 的 hidden state
        self._hooks = []

    def _register_hooks(self):
        """注册 forward hook 采集每层输出."""
        self.hidden_states = []
        self._hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output 是 tuple (hidden, ...) 或 tensor
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # 只存最后一个 token 位置, [hidden_dim]
                last_token_hidden = hidden[0, -1, :].detach().cpu()
                self.hidden_states.append((layer_idx, last_token_hidden))
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
                            primary_answer_ids: List[int]) -> Dict:
        """对一个样本跑一次 forward, 采集每层轨迹.

        Returns:
            {
                "logit_lens_logits_per_layer": List[Tensor],  # 每层 logit lens 后的 logits
                "correct_token_logprob": List[float],   # 每层正确答案 token 的 log prob
                "correct_token_rank": List[int],        # 每层正确答案 token 的 rank
                "top5_per_layer": List[List[(token_id, prob)]],
            }
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 1. 注册 hook
        self._register_hooks()
        self.hidden_states = []

        # 2. forward (只前向, 不生成)
        outputs = self.model(input_ids, use_cache=False, output_hidden_states=False)

        # 3. 移除 hook
        self._remove_hooks()

        # hidden_states 现在是 [(layer_idx, [hidden_dim]), ...]
        # 按层排序
        self.hidden_states.sort(key=lambda x: x[0])

        # 4. 对每层 hidden state 用 logit lens 解码
        target_token = primary_answer_ids[0]
        correct_logprob_per_layer = []
        correct_rank_per_layer = []
        top5_per_layer = []

        # 加上 embedding 层 (layer 0 = embedding output)
        # outputs.hidden_states 包含 embedding + 每层输出
        # 我们也加上 embedding 层
        # 但 hook 只采了 transformer block 的, 我们单独取 embedding
        # 简化: 只用 hook 采集的层

        for layer_idx, hidden in self.hidden_states:
            # hidden: [hidden_dim]
            # reshape 成 [1, 1, hidden_dim] 以适应 logit_lens_decode
            h = hidden.unsqueeze(0).unsqueeze(0).to(device).to(self.unembed.dtype)
            # 送到和 unembed 同一设备
            # final_norm 和 unembed 都在 model 设备上
            # 但 hidden 已经 .cpu() 了, 要送回去
            h = h.to(self.unembed.device)

            with torch.no_grad():
                # 应用最终 norm + unembedding
                normed = self.final_norm(h)
                logits = F.linear(normed, self.unembed)  # [1, 1, vocab]
                logits = logits.squeeze(0).squeeze(0)  # [vocab]

            # log softmax
            log_probs = F.log_softmax(logits, dim=-1)
            correct_logprob = log_probs[target_token].item()
            correct_logprob_per_layer.append(correct_logprob)

            # rank
            sorted_idx = torch.argsort(logits, descending=True)
            rank = (sorted_idx == target_token).nonzero(as_tuple=True)[0].item()
            correct_rank_per_layer.append(rank)

            # top 5
            topk_vals, topk_idx = torch.topk(logits, k=5)
            topk_probs = F.softmax(topk_vals, dim=-1)
            top5 = list(zip(topk_idx.tolist(), topk_probs.tolist()))
            top5_per_layer.append(top5)

        return {
            "logit_lens_logits_per_layer": None,  # 不存原始 logits, 省内存
            "correct_token_logprob": correct_logprob_per_layer,
            "correct_token_rank": correct_rank_per_layer,
            "top5_per_layer": top5_per_layer,
            "num_layers_collected": len(correct_logprob_per_layer),
        }

    @torch.no_grad()
    def generate_answer(self, prompt_ids: List[int]) -> str:
        """生成答案, 用于判断最终是否答对."""
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        out = self.model.generate(
            input_ids,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=config.DO_SAMPLE,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        generated = out[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text
