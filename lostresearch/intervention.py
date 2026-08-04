"""
第五章: 因果干预 - Activation Patching

三个实验:
1. Necessity: 消融正确答案方向, 看 CIS 是否下降
2. Clean-to-Corrupted: 把 clean run 的激活复制到 corrupted run
3. Prediction-Guided: 在 high-decay-risk vs low-decay-risk 样本上做 patching

关键: 修改 forward pass, 在指定层/位置注入激活.
"""
import json
import os
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config


def get_residual_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Cannot find layers")


class ActivationPatcher:
    """Activation patching 工具.

    用法:
        patcher = ActivationPatcher(model)
        # 1. 跑 clean run, 记录每层 hidden
        clean_hiddens = patcher.collect_hiddens(clean_prompt_ids)
        # 2. 跑 corrupted run, 在第 l 层注入 clean 的 hidden
        result = patcher.patched_forward(corr_prompt_ids, clean_hiddens,
                                          patch_layer=l, patch_pos=-1)
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = get_residual_layers(model)
        self.num_layers = len(self.layers)
        self._patch_hook = None
        self._patch_value = None
        self._patch_layer = None
        self._patch_pos = None

    def _clear_patch(self):
        if self._patch_hook is not None:
            self._patch_hook.remove()
            self._patch_hook = None
        self._patch_value = None
        self._patch_layer = None
        self._patch_pos = None

    @torch.no_grad()
    def collect_hiddens(self, prompt_ids: List[int]) -> Dict[int, torch.Tensor]:
        """跑一次 forward, 记录每层 hidden state. 返回 {layer_idx: [seq, hidden]}."""
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        hiddens = {}
        hooks = []

        def make_hook(idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                hiddens[idx] = h[0].detach().clone()  # [seq, hidden]
            return hook

        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        self.model(input_ids, use_cache=False)

        for h in hooks:
            h.remove()

        return hiddens

    @staticmethod
    def _find_common_prefix_len(a: List[int], b: List[int]) -> int:
        """找到两个 token 序列的最长公共前缀长度."""
        n = 0
        for x, y in zip(a, b):
            if x == y:
                n += 1
            else:
                break
        return n

    @torch.no_grad()
    def patched_forward(self, prompt_ids: List[int],
                         clean_hiddens: Dict[int, torch.Tensor],
                         patch_layer: int,
                         patch_pos: int = -1,
                         clean_prompt_ids: List[int] = None,
                         target_token_id: int = None) -> Dict:
        """跑 corrupted run, 在 patch_layer 的 patch_pos 位置注入 clean 的 hidden.

        Args:
            prompt_ids: corrupted prompt
            clean_hiddens: clean run 的 {layer: [seq, hidden]}
            patch_layer: 在哪层 patch
            patch_pos: 在哪个位置 patch (-1 = 最后一个公共 token)
            clean_prompt_ids: clean prompt 的 token ids (用于计算公共前缀)
            target_token_id: 要测的正确答案 token

        Returns:
            {logit, target_logprob, target_rank, top5}
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        seq_len = len(prompt_ids)

        if patch_pos < 0 and clean_prompt_ids is not None:
            common_len = self._find_common_prefix_len(clean_prompt_ids, prompt_ids)
            actual_pos = max(0, common_len - 1)
        elif patch_pos >= 0:
            actual_pos = patch_pos
        else:
            actual_pos = seq_len - 1

        clean_seq_len = clean_hiddens[patch_layer].shape[0]
        if actual_pos >= clean_seq_len:
            actual_pos = clean_seq_len - 1
        # 保险: 确保 actual_pos 不超过 corrupted run 的 seq_len
        if actual_pos >= seq_len:
            actual_pos = seq_len - 1
        if actual_pos < 0:
            actual_pos = 0

        self._patch_value = clean_hiddens[patch_layer][actual_pos].clone()
        self._patch_layer = patch_layer
        self._patch_pos = actual_pos

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            # h 可能是 [batch, seq, hidden] 或 [batch, hidden]
            if h.dim() == 3:
                if h.dim() == 3:
                h[0, actual_pos] = self._patch_value.to(h.device).to(h.dtype)
            elif h.dim() == 2:
                h[0] = self._patch_value.to(h.device).to(h.dtype)
            else:
                raise ValueError(f"Unexpected h dim: {h.dim()}")
            elif h.dim() == 2:
                h[0] = self._patch_value.to(h.device).to(h.dtype)
            else:
                raise ValueError(f"Unexpected h dim: {h.dim()}")
            if rest:
                return (h,) + rest
            return h

        self._patch_hook = self.layers[patch_layer].register_forward_hook(patch_hook)

        try:
            outputs = self.model(input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :]
        finally:
            self._clear_patch()

        log_probs = F.log_softmax(logits, dim=-1)
        result = {
            "logits": logits.cpu(),
            "top5": torch.topk(logits, 5).indices.tolist(),
        }
        if target_token_id is not None and target_token_id < logits.shape[-1]:
            result["target_logprob"] = log_probs[target_token_id].item()
            sorted_idx = torch.argsort(logits, descending=True)
            result["target_rank"] = (sorted_idx == target_token_id).nonzero(as_tuple=True)[0].item()
            result["target_in_top5"] = target_token_id in result["top5"]
        return result

    @torch.no_grad()
    def patched_forward_multi_pos(self, prompt_ids: List[int],
                                    clean_hiddens: Dict[int, torch.Tensor],
                                    patch_layer: int,
                                    clean_prompt_ids: List[int],
                                    target_token_id: int = None) -> Dict:
        """Patch ALL positions in the common prefix (not just one).

        这是对单位置 patch 的改进: 将 clean run 的整个公共前缀表示注入到
        corrupted run 中, 而不是只注入一个 token 位置.
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        common_len = self._find_common_prefix_len(clean_prompt_ids, prompt_ids)
        clean_seq_len = clean_hiddens[patch_layer].shape[0]
        corr_seq_len = len(prompt_ids)
        patch_len = min(common_len, clean_seq_len, corr_seq_len)

        if patch_len == 0:
            return self.patched_forward(prompt_ids, clean_hiddens, patch_layer,
                                         clean_prompt_ids=clean_prompt_ids,
                                         target_token_id=target_token_id)

        self._patch_layer = patch_layer

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            for pos in range(patch_len):
                if h.dim() == 3:
                    h[0, pos] = clean_hiddens[patch_layer][pos].to(h.device).to(h.dtype)
                elif h.dim() == 2:
                    h[0] = clean_hiddens[patch_layer][pos].to(h.device).to(h.dtype)
                else:
                    raise ValueError(f"Unexpected h dim: {h.dim()}")
            if rest:
                return (h,) + rest
            return h

        self._patch_hook = self.layers[patch_layer].register_forward_hook(patch_hook)

        try:
            outputs = self.model(input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :]
        finally:
            self._clear_patch()

        log_probs = F.log_softmax(logits, dim=-1)
        result = {
            "logits": logits.cpu(),
            "top5": torch.topk(logits, 5).indices.tolist(),
        }
        if target_token_id is not None and target_token_id < logits.shape[-1]:
            result["target_logprob"] = log_probs[target_token_id].item()
            sorted_idx = torch.argsort(logits, descending=True)
            result["target_rank"] = (sorted_idx == target_token_id).nonzero(as_tuple=True)[0].item()
            result["target_in_top5"] = target_token_id in result["top5"]
        return result

    @torch.no_grad()
    def patched_forward_last_pos(self, prompt_ids: List[int],
                                   clean_hiddens: Dict[int, torch.Tensor],
                                   patch_layer: int,
                                   target_token_id: int = None) -> Dict:
        """Patch the last token position (用于等长 corrupted prompt).

        Clean 和 corrupted 序列长度相同时, 直接 patch 最后一个 token 位置,
        这是答案生成的位置, 应该是最有效的 patch 位置.
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        seq_len = len(prompt_ids)
        clean_seq_len = clean_hiddens[patch_layer].shape[0]
        actual_pos = min(seq_len - 1, clean_seq_len - 1)

        self._patch_value = clean_hiddens[patch_layer][actual_pos].clone()
        self._patch_layer = patch_layer
        self._patch_pos = actual_pos

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            if h.dim() == 3:
                h[0, actual_pos] = self._patch_value.to(h.device).to(h.dtype)
            elif h.dim() == 2:
                h[0] = self._patch_value.to(h.device).to(h.dtype)
            else:
                raise ValueError(f"Unexpected h dim: {h.dim()}")
            if rest:
                return (h,) + rest
            return h

        self._patch_hook = self.layers[patch_layer].register_forward_hook(patch_hook)

        try:
            outputs = self.model(input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :]
        finally:
            self._clear_patch()

        log_probs = F.log_softmax(logits, dim=-1)
        result = {
            "logits": logits.cpu(),
            "top5": torch.topk(logits, 5).indices.tolist(),
        }
        if target_token_id is not None and target_token_id < logits.shape[-1]:
            result["target_logprob"] = log_probs[target_token_id].item()
            sorted_idx = torch.argsort(logits, descending=True)
            result["target_rank"] = (sorted_idx == target_token_id).nonzero(as_tuple=True)[0].item()
            result["target_in_top5"] = target_token_id in result["top5"]
        return result

    @torch.no_grad()
    def patched_generate(self, prompt_ids: List[int],
                          clean_hiddens: Dict[int, torch.Tensor],
                          patch_layer: int,
                          clean_prompt_ids: List[int] = None,
                          max_new_tokens: int = 32) -> str:
        """Patched forward + generate: 看 patching 后生成的答案是否变化.

        这是 gold standard 因果测试: 不只测 logprob, 而是实际生成文本.
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        if clean_prompt_ids is not None:
            common_len = self._find_common_prefix_len(clean_prompt_ids, prompt_ids)
            clean_seq_len = clean_hiddens[patch_layer].shape[0]
            corr_seq_len = len(prompt_ids)
            patch_len = min(common_len, clean_seq_len, corr_seq_len)
        else:
            patch_len = min(len(prompt_ids), clean_hiddens[patch_layer].shape[0])

        self._patch_layer = patch_layer
        self._patch_len = patch_len

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            for pos in range(patch_len):
                if h.dim() == 3:
                    h[0, pos] = clean_hiddens[patch_layer][pos].to(h.device).to(h.dtype)
                elif h.dim() == 2:
                    h[0] = clean_hiddens[patch_layer][pos].to(h.device).to(h.dtype)
                else:
                    raise ValueError(f"Unexpected h dim: {h.dim()}")
            if rest:
                return (h,) + rest
            return h

        self._patch_hook = self.layers[patch_layer].register_forward_hook(patch_hook)

        try:
            gen_out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = gen_out[0][input_ids.shape[1]:]
            generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        finally:
            self._clear_patch()

        return generated.strip()

    @torch.no_grad()
    def patched_forward_with_logit_lens(self, prompt_ids: List[int],
                                          clean_hiddens: Dict[int, torch.Tensor],
                                          patch_layer: int,
                                          patch_pos: int = -1,
                                          target_token_id: int = None) -> Dict:
        """Patched forward + 每层 logit lens, 看注入后后续层的 CIS 变化."""
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        seq_len = len(prompt_ids)
        actual_pos = patch_pos if patch_pos >= 0 else seq_len - 1

        self._patch_value = clean_hiddens[patch_layer][actual_pos].clone()
        self._patch_layer = patch_layer
        self._patch_pos = actual_pos

        hiddens = {}
        hooks = []

        def make_hook(idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                hiddens[idx] = h[0].detach().clone()
            return hook

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            if h.dim() == 3:
                h[0, actual_pos] = self._patch_value.to(h.device).to(h.dtype)
            elif h.dim() == 2:
                h[0] = self._patch_value.to(h.device).to(h.dtype)
            else:
                raise ValueError(f"Unexpected h dim: {h.dim()}")
            if rest:
                return (h,) + rest
            return h

        self._patch_hook = self.layers[patch_layer].register_forward_hook(patch_hook)
        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        try:
            outputs = self.model(input_ids, use_cache=False)
        finally:
            self._clear_patch()
            for h in hooks:
                h.remove()

        # 最终层 logits
        final_logits = outputs.logits[0, -1, :]
        log_probs = F.log_softmax(final_logits, dim=-1)
        result = {
            "final_logits": final_logits.cpu(),
            "hiddens": {k: v.cpu() for k, v in hiddens.items()},
        }
        if target_token_id is not None and target_token_id < final_logits.shape[-1]:
            result["target_logprob"] = log_probs[target_token_id].item()
            sorted_idx = torch.argsort(final_logits, descending=True)
            result["target_rank"] = (sorted_idx == target_token_id).nonzero(as_tuple=True)[0].item()
        return result


def create_corrupted_prompt(original_question: str, tokenizer) -> str:
    """把问题中的关键实体替换掉, 制造 corrupted prompt.

    关键: 保持 prompt 长度与 clean 一致, 避免 patching 时索引越界.
    策略: 在问题末尾加误导性 context, 然后截断到与 clean prompt 相同长度.
    """
    misleading = "\nNote: the answer might be different from what you think."
    messages = [{"role": "user", "content": original_question + misleading + " Answer with just the answer."}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING,
    )


def create_corrupted_prompt_same_len(original_question: str, wrong_answer: str, tokenizer) -> str:
    """创建等长 corrupted prompt: 替换问题中的关键信息而非追加.

    策略: 把问题本身改成问一个不同但相关的问题, 保持 prompt 长度相同.
    例如: "What is the capital of France?" → "What is the capital of Germany?"
    这样 clean 和 corrupted 的 token 序列长度完全相同, 可以 patch 最后一个 token.
    """
    clean_messages = [{"role": "user", "content": original_question + " Answer with just the answer."}]
    clean_prompt = tokenizer.apply_chat_template(
        clean_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING,
    )

    corrupted_messages = [{"role": "user", "content": original_question + "\nHint: the answer is " + wrong_answer + ". Answer with just the answer."}]
    corrupted_prompt = tokenizer.apply_chat_template(
        corrupted_messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING,
    )

    return corrupted_prompt


def run_necessity_experiment(patcher: ActivationPatcher,
                              samples: List[Dict],
                              n_samples: int = 20) -> List[Dict]:
    """实验 5.1: Necessity - 消融正确答案方向, 看 CIS 是否下降.

    对每个样本:
    1. 正常 forward, 记录每层 hidden
    2. 找到与正确答案 token 最相关的方向 (用 unembedding 的对应列)
    3. 在每层消融这个方向 (把 hidden 投影到该方向的补空间)
    4. 看 CIS 和最终 logit 是否下降
    """
    print("\n  [5.1] Necessity: 消融正确答案方向")
    results = []

    # 获取 unembedding
    unembed = patcher.model.lm_head.weight  # [vocab, hidden]

    for i, s in enumerate(tqdm(samples[:n_samples], desc="Necessity")):
        try:
            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            device = patcher.model.device

            # 1. 正常 forward
            normal_out = patcher.model(
                torch.tensor([prompt_ids], dtype=torch.long, device=device),
                use_cache=False
            )
            normal_logits = normal_out.logits[0, -1, :]
            normal_log_probs = F.log_softmax(normal_logits, dim=-1)
            normal_target_lp = normal_log_probs[target_token].item()

            # 2. 正确答案方向 = unembedding 的对应列
            answer_dir = unembed[target_token].detach()  # [hidden]

            # 3. 在每层消融这个方向 (投影到补空间)
            ablation_results = []
            for layer_idx in range(patcher.num_layers):
                # 用 hook 消融
                def ablate_hook(module, input, output, layer_idx=layer_idx):
                    if isinstance(output, tuple):
                        h = output[0]
                        rest = output[1:]
                    else:
                        h = output
                        rest = ()
                    # 消融最后一个 token 位置的 answer_dir 方向
                    last_h = h[0, -1, :]  # [hidden]
                    proj = (last_h @ answer_dir) / (answer_dir @ answer_dir) * answer_dir
                    h[0, -1, :] = last_h - proj
                    if rest:
                        return (h,) + rest
                    return h

                h = patcher.layers[layer_idx].register_forward_hook(ablate_hook)
                ablated_out = patcher.model(
                    torch.tensor([prompt_ids], dtype=torch.long, device=device),
                    use_cache=False
                )
                h.remove()

                ablated_logits = ablated_out.logits[0, -1, :]
                ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)
                ablated_target_lp = ablated_log_probs[target_token].item()
                ablation_results.append({
                    "layer": layer_idx,
                    "target_logprob": ablated_target_lp,
                    "delta": normal_target_lp - ablated_target_lp,  # 正: 消融后下降
                })

            results.append({
                "sample_id": s["id"],
                "question": s["question"][:50],
                "answer": s["answer"],
                "normal_target_logprob": normal_target_lp,
                "ablation_per_layer": ablation_results,
            })

            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{n_samples}")

        except Exception as e:
            print(f"    Error on {s['id']}: {e}")
            continue

    return results


def run_patch_experiment(patcher: ActivationPatcher,
                          samples: List[Dict],
                          all_results: List[Dict],
                          n_samples: int = 20) -> List[Dict]:
    """实验 5.2 + 5.3: Clean-to-Corrupted Patching (改进版).

    三种 patching 策略:
      A. 单位置: patch 公共前缀最后一个 token (原版, 基线)
      B. 多位置: patch 公共前缀的所有 token (改进1)
      C. 等长末位: 等长 corrupted prompt, patch 最后一个 token (改进2)
      D. 生成测试: patch 后实际 generate, 看答案是否被纠正 (改进3)

    只在关键层 (31-35) 做 patching, 减少计算量.
    """
    print("\n  [5.2+5.3] Patching (3 strategies, key layers only)")
    results = []

    key_layers = [31, 32, 33, 34, 35]

    for i, s in enumerate(tqdm(samples[:n_samples], desc="Patching")):
        try:
            result = next((r for r in all_results if r["id"] == s["id"]), None)
            if result is None:
                continue

            cis = result["cis"]
            n_cis = len(cis)
            if n_cis < 4:
                continue
            t0_idx = max(2, int(n_cis * config.PREDICTION_T0))
            cis_at_t0 = cis[t0_idx]
            cis_final = cis[-1]
            high_risk = (cis_at_t0 > 0 and cis_final < 0)

            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            device = patcher.model.device

            clean_hiddens = patcher.collect_hiddens(prompt_ids)
            if i == 0:
                sample_layer_shape = clean_hiddens[0].shape
                print(f"    [diag] clean_hiddens[0].shape={sample_layer_shape}, "
                      f"prompt_len={len(prompt_ids)}, corr_ids_len={len(corr_ids) if 'corr_ids' in dir() else '?'}")
            clean_out = patcher.model(
                torch.tensor([prompt_ids], dtype=torch.long, device=device),
                use_cache=False
            )
            clean_log_probs = F.log_softmax(clean_out.logits[0, -1, :], dim=-1)
            clean_target_lp = clean_log_probs[target_token].item()

            # === Strategy A: 单位置 patch (原版) ===
            corr_text = create_corrupted_prompt(s["question"], patcher.tokenizer)
            corr_ids = patcher.tokenizer.encode(corr_text, add_special_tokens=False)
            corr_ids = corr_ids[-config.MAX_PROMPT_LEN:]

            corr_out = patcher.model(
                torch.tensor([corr_ids], dtype=torch.long, device=device),
                use_cache=False
            )
            corr_log_probs = F.log_softmax(corr_out.logits[0, -1, :], dim=-1)
            corr_target_lp = corr_log_probs[target_token].item()

            strategy_a = []
            for layer_idx in key_layers:
                patched = patcher.patched_forward(
                    corr_ids, clean_hiddens,
                    patch_layer=layer_idx, patch_pos=-1,
                    clean_prompt_ids=prompt_ids,
                    target_token_id=target_token,
                )
                strategy_a.append({
                    "layer": layer_idx,
                    "target_logprob": patched["target_logprob"],
                    "target_rank": patched["target_rank"],
                })

            # === Strategy B: 多位置 patch ===
            strategy_b = []
            for layer_idx in key_layers:
                patched = patcher.patched_forward_multi_pos(
                    corr_ids, clean_hiddens,
                    patch_layer=layer_idx,
                    clean_prompt_ids=prompt_ids,
                    target_token_id=target_token,
                )
                strategy_b.append({
                    "layer": layer_idx,
                    "target_logprob": patched["target_logprob"],
                    "target_rank": patched["target_rank"],
                })

            # === Strategy C: 等长 corrupted + last-pos patch ===
            wrong_answer = s.get("wrong_answer", "unknown")
            corr_same_text = create_corrupted_prompt_same_len(
                s["question"], wrong_answer, patcher.tokenizer
            )
            corr_same_ids = patcher.tokenizer.encode(corr_same_text, add_special_tokens=False)
            corr_same_ids = corr_same_ids[-config.MAX_PROMPT_LEN:]

            corr_same_out = patcher.model(
                torch.tensor([corr_same_ids], dtype=torch.long, device=device),
                use_cache=False
            )
            corr_same_log_probs = F.log_softmax(corr_same_out.logits[0, -1, :], dim=-1)
            corr_same_target_lp = corr_same_log_probs[target_token].item()

            strategy_c = []
            for layer_idx in key_layers:
                patched = patcher.patched_forward_last_pos(
                    corr_same_ids, clean_hiddens,
                    patch_layer=layer_idx,
                    target_token_id=target_token,
                )
                strategy_c.append({
                    "layer": layer_idx,
                    "target_logprob": patched["target_logprob"],
                    "target_rank": patched["target_rank"],
                })

            # === Strategy D: 生成测试 (只测最佳层 35) ===
            gen_clean = patcher.tokenizer.decode(
                patcher.model.generate(
                    torch.tensor([prompt_ids], dtype=torch.long, device=device),
                    max_new_tokens=32, do_sample=False,
                    pad_token_id=patcher.tokenizer.eos_token_id,
                )[0][len(prompt_ids):], skip_special_tokens=True
            ).strip()

            gen_corr = patcher.tokenizer.decode(
                patcher.model.generate(
                    torch.tensor([corr_ids], dtype=torch.long, device=device),
                    max_new_tokens=32, do_sample=False,
                    pad_token_id=patcher.tokenizer.eos_token_id,
                )[0][len(corr_ids):], skip_special_tokens=True
            ).strip()

            gen_patched = patcher.patched_generate(
                corr_ids, clean_hiddens, patch_layer=35,
                clean_prompt_ids=prompt_ids, max_new_tokens=32,
            )

            gen_corr_same = patcher.tokenizer.decode(
                patcher.model.generate(
                    torch.tensor([corr_same_ids], dtype=torch.long, device=device),
                    max_new_tokens=32, do_sample=False,
                    pad_token_id=patcher.tokenizer.eos_token_id,
                )[0][len(corr_same_ids):], skip_special_tokens=True
            ).strip()

            gen_patched_same = patcher.patched_generate(
                corr_same_ids, clean_hiddens, patch_layer=35,
                clean_prompt_ids=None, max_new_tokens=32,
            )

            results.append({
                "sample_id": s["id"],
                "question": s["question"][:80],
                "answer": s["answer"],
                "final_correct": result["final_correct"],
                "high_risk": high_risk,
                "clean_target_logprob": clean_target_lp,
                "corrupted_target_logprob": corr_target_lp,
                "corrupted_same_target_logprob": corr_same_target_lp,
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "strategy_c": strategy_c,
                "gen_clean": gen_clean,
                "gen_corr": gen_corr,
                "gen_patched": gen_patched,
                "gen_corr_same": gen_corr_same,
                "gen_patched_same": gen_patched_same,
            })

            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{n_samples}")

        except Exception as e:
            print(f"    Error on {s['id']}: {e}")
            continue

    return results


def analyze_intervention_results(necessity_results: List[Dict],
                                   patch_results: List[Dict]) -> Dict:
    """分析因果干预结果."""
    print("\n" + "=" * 70)
    print("CAUSAL INTERVENTION ANALYSIS")
    print("=" * 70)

    # 1. Necessity 分析
    if necessity_results:
        print("\n--- 5.1 Necessity ---")
        # 找哪些层消融影响最大
        all_deltas = []
        for r in necessity_results:
            for al in r["ablation_per_layer"]:
                all_deltas.append((al["layer"], al["delta"]))

        # 按层平均
        layer_deltas = {}
        for l, d in all_deltas:
            if l not in layer_deltas:
                layer_deltas[l] = []
            layer_deltas[l].append(d)

        print("  消融后 target logprob 下降最多的层 (top 5):")
        layer_means = [(l, np.mean(ds)) for l, ds in layer_deltas.items()]
        layer_means.sort(key=lambda x: -x[1])
        for l, d in layer_means[:5]:
            print(f"    Layer {l}: delta={d:.2f} (正=消融后下降)")

        print(f"\n  总样本数: {len(necessity_results)}")
        has_effect = sum(1 for r in necessity_results
                        if any(al["delta"] > 1.0 for al in r["ablation_per_layer"]))
        print(f"  至少一层消融有效 (delta>1): {has_effect}/{len(necessity_results)}")

    # 2. Patching 分析
    if patch_results:
        print("\n--- 5.2 Patching: Logprob Recovery ---")

        for strategy_name, strategy_key in [
            ("Strategy A (单位置, 前缀)", "strategy_a"),
            ("Strategy B (多位置, 前缀)", "strategy_b"),
            ("Strategy C (等长, 末位)", "strategy_c"),
        ]:
            all_recoveries = []
            for r in patch_results:
                if strategy_key not in r:
                    continue
                corr_key = "corrupted_target_logprob" if "same" not in strategy_key else "corrupted_same_target_logprob"
                corr_lp = r.get(corr_key, r.get("corrupted_target_logprob", 0))
                for pl in r[strategy_key]:
                    recovery = pl["target_logprob"] - corr_lp
                    all_recoveries.append((pl["layer"], recovery, r["high_risk"]))

            if not all_recoveries:
                continue

            layer_recoveries = {}
            for l, rec, hr in all_recoveries:
                if l not in layer_recoveries:
                    layer_recoveries[l] = []
                layer_recoveries[l].append(rec)

            layer_means = [(l, np.mean(v)) for l, v in layer_recoveries.items()]
            layer_means.sort(key=lambda x: -x[1])
            best_layer, best_rec = layer_means[0]
            print(f"  {strategy_name}:")
            print(f"    最佳层: Layer {best_layer}, recovery={best_rec:.4f}")

            high_recs = [rec for _, rec, hr in all_recoveries if hr]
            low_recs = [rec for _, rec, hr in all_recoveries if not hr]
            if high_recs and low_recs:
                print(f"    High-risk mean recovery: {np.mean(high_recs):.4f}")
                print(f"    Low-risk mean recovery:  {np.mean(low_recs):.4f}")

        # 3. Generation test
        print("\n--- 5.3 Generation Test (Gold Standard) ---")
        print("  (clean answer → corrupted answer → patched answer)")

        n_changed = 0
        n_reverted = 0
        n_corrected = 0
        n_total = 0
        examples = []

        for r in patch_results:
            if "gen_clean" not in r:
                continue
            n_total += 1
            clean_ans = r["gen_clean"].strip()
            corr_ans = r["gen_corr"].strip()
            patched_ans = r["gen_patched"].strip()
            corr_same_ans = r.get("gen_corr_same", "").strip()
            patched_same_ans = r.get("gen_patched_same", "").strip()

            if corr_ans != clean_ans:
                n_changed += 1
            if patched_ans == clean_ans and corr_ans != clean_ans:
                n_reverted += 1
                examples.append((r["sample_id"], r["question"][:60], clean_ans, corr_ans, patched_ans, "multi-pos"))
            if patched_same_ans == clean_ans and corr_same_ans != clean_ans:
                n_corrected += 1
                examples.append((r["sample_id"], r["question"][:60], clean_ans, corr_same_ans, patched_same_ans, "same-len"))

        print(f"  Corrupted prompt 改变了答案: {n_changed}/{n_total}")
        print(f"  多位置 patch 后答案恢复: {n_reverted}/{n_total}")
        print(f"  等长 patch 后答案恢复: {n_corrected}/{n_total}")

        if examples:
            print(f"\n  恢复成功示例 (前5个):")
            for sid, q, ca, cra, pa, st in examples[:5]:
                print(f"    [{st}] {sid}: '{ca}' → corr:'{cra}' → patched:'{pa}'")
                print(f"      Q: {q}")

        if n_reverted == 0 and n_corrected == 0:
            print("  ⚠ 两种策略均未恢复答案 → 说明信号是分布式的, 需要 patch 更多层")

    return {
        "necessity_analyzed": len(necessity_results),
        "patching_analyzed": len(patch_results),
    }


def run_intervention_experiment(model, tokenizer,
                                  prepared_samples: List[Dict],
                                  all_results: List[Dict]) -> Dict:
    """运行完整的因果干预实验."""
    print("\n" + "=" * 70)
    print("CHAPTER 5: CAUSAL INTERVENTION")
    print("=" * 70)

    patcher = ActivationPatcher(model, tokenizer)

    # 5.1 Necessity
    print("\n[5.1] Necessity Experiment")
    necessity_results = run_necessity_experiment(patcher, prepared_samples, n_samples=50)

    # 5.2 + 5.3 Patching
    print("\n[5.2+5.3] Patching Experiment")
    patch_results = run_patch_experiment(patcher, prepared_samples, all_results, n_samples=50)

    # 分析
    analysis = analyze_intervention_results(necessity_results, patch_results)

    return {
        "necessity": necessity_results,
        "patching": patch_results,
        "analysis": analysis,
    }