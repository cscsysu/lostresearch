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

    @torch.no_grad()
    def patched_forward(self, prompt_ids: List[int],
                         clean_hiddens: Dict[int, torch.Tensor],
                         patch_layer: int,
                         patch_pos: int = -1,
                         target_token_id: int = None) -> Dict:
        """跑 corrupted run, 在 patch_layer 的 patch_pos 位置注入 clean 的 hidden.

        Args:
            prompt_ids: corrupted prompt
            clean_hiddens: clean run 的 {layer: [seq, hidden]}
            patch_layer: 在哪层 patch
            patch_pos: 在哪个位置 patch (-1 = 最后一个 token)
            target_token_id: 要测的正确答案 token

        Returns:
            {logit, target_logprob, target_rank, top5}
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        seq_len = len(prompt_ids)
        actual_pos = patch_pos if patch_pos >= 0 else seq_len - 1

        # 准备 patch 值
        self._patch_value = clean_hiddens[patch_layer][actual_pos].clone()  # [hidden]
        self._patch_layer = patch_layer
        self._patch_pos = actual_pos

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = ()
            # 替换指定位置
            h[0, actual_pos] = self._patch_value.to(h.device).to(h.dtype)
            if rest:
                return (h,) + rest
            return h

        self._patch_hook = self.layers[patch_layer].register_forward_hook(patch_hook)

        try:
            outputs = self.model(input_ids, use_cache=False)
            logits = outputs.logits[0, -1, :]  # [vocab]
        finally:
            self._clear_patch()

        log_probs = F.log_softmax(logits, dim=-1)
        result = {
            "logits": logits.cpu(),
            "top5": list(zip(*torch.topk(logits, 5)))[0].tolist(),
        }
        if target_token_id is not None and target_token_id < logits.shape[-1]:
            result["target_logprob"] = log_probs[target_token_id].item()
            sorted_idx = torch.argsort(logits, descending=True)
            result["target_rank"] = (sorted_idx == target_token_id).nonzero(as_tuple=True)[0].item()
            result["target_in_top5"] = target_token_id in result["top5"]
        return result

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
            h[0, actual_pos] = self._patch_value.to(h.device).to(h.dtype)
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
    """把问题中的关键实体替换掉, 制造 corrupted prompt."""
    # 简单策略: 在问题末尾加一个误导性的 context
    # 更好的方法是用 NER 找实体, 但 pilot 阶段先简单做
    misleading = "\nNote: the answer might be different from what you think."
    messages = [{"role": "user", "content": original_question + misleading + " Answer with just the answer."}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=config.ENABLE_THINKING,
    )


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
    """实验 5.2 + 5.3: Clean-to-Corrupted Patching + Prediction-Guided.

    对每个样本:
    1. Clean run: 原始 prompt
    2. Corrupted run: 加误导 context
    3. 在每层做 patching: 把 clean 的 hidden 注入 corrupted
    4. 看 patching 后正确答案 logit 是否恢复

    同时根据预测器分组 (high-decay-risk vs low-decay-risk).
    """
    print("\n  [5.2+5.3] Clean-to-Corrupted Patching + Prediction-Guided")
    results = []

    for i, s in enumerate(tqdm(samples[:n_samples], desc="Patching")):
        try:
            # 找到对应的 result
            result = next((r for r in all_results if r["id"] == s["id"]), None)
            if result is None:
                continue

            # 判断 high/low decay risk
            cis = result["cis"]
            n = len(cis)
            if n < 4:
                continue
            t0_idx = max(2, int(n * config.PREDICTION_T0))
            cis_at_t0 = cis[t0_idx]
            cis_final = cis[-1]
            # high-risk: 中间 CIS>0 但最终 CIS<0
            high_risk = (cis_at_t0 > 0 and cis_final < 0)

            prompt_ids = s["prompt_ids"]
            target_token = s["primary_answer_ids"][0]
            device = patcher.model.device

            # 1. Clean run
            clean_hiddens = patcher.collect_hiddens(prompt_ids)

            # 2. Corrupted prompt (加误导)
            corr_prompt_text = create_corrupted_prompt(s["question"], patcher.tokenizer)
            corr_prompt_ids = patcher.tokenizer.encode(corr_prompt_text, add_special_tokens=False)
            corr_prompt_ids = corr_prompt_ids[-config.MAX_PROMPT_LEN:]

            # 3. 在每层做 patching
            patching_results = []
            for layer_idx in range(min(patcher.num_layers, 36)):  # 限制层数省时间
                patched = patcher.patched_forward(
                    corr_prompt_ids, clean_hiddens,
                    patch_layer=layer_idx,
                    patch_pos=-1,
                    target_token_id=target_token,
                )
                patching_results.append({
                    "layer": layer_idx,
                    "target_logprob": patched["target_logprob"],
                    "target_rank": patched["target_rank"],
                })

            # 4. 不 patching 的 baseline (corrupted run 原始)
            corr_out = patcher.model(
                torch.tensor([corr_prompt_ids], dtype=torch.long, device=device),
                use_cache=False
            )
            corr_logits = corr_out.logits[0, -1, :]
            corr_log_probs = F.log_softmax(corr_logits, dim=-1)
            corr_target_lp = corr_log_probs[target_token].item()

            results.append({
                "sample_id": s["id"],
                "question": s["question"][:50],
                "answer": s["answer"],
                "final_correct": result["final_correct"],
                "high_risk": high_risk,
                "clean_target_logprob": result["correct_logprob"][-1],  # clean run 的最终层
                "corrupted_target_logprob": corr_target_lp,
                "patching_per_layer": patching_results,
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
        print("\n--- 5.2+5.3 Patching ---")
        # 找哪些层 patching 恢复最多
        all_recoveries = []
        for r in patch_results:
            corr_lp = r["corrupted_target_logprob"]
            for pl in r["patching_per_layer"]:
                recovery = pl["target_logprob"] - corr_lp
                all_recoveries.append((pl["layer"], recovery, r["high_risk"]))

        # 按层平均
        layer_recoveries = {}
        for l, rec, hr in all_recoveries:
            if l not in layer_recoveries:
                layer_recoveries[l] = {"all": [], "high": [], "low": []}
            layer_recoveries[l]["all"].append(rec)
            if hr:
                layer_recoveries[l]["high"].append(rec)
            else:
                layer_recoveries[l]["low"].append(rec)

        print("  Patching 恢复效果最好的层 (top 5):")
        layer_means = [(l, np.mean(v["all"])) for l, v in layer_recoveries.items()]
        layer_means.sort(key=lambda x: -x[1])
        for l, r in layer_means[:5]:
            print(f"    Layer {l}: recovery={r:.2f}")

        # 3. Prediction-Guided 对比
        high_risk = [r for r in patch_results if r["high_risk"]]
        low_risk = [r for r in patch_results if not r["high_risk"]]
        print(f"\n--- 5.3 Prediction-Guided ---")
        print(f"  High-risk (中间CIS>0, 最终CIS<0): {len(high_risk)}")
        print(f"  Low-risk: {len(low_risk)}")

        if high_risk and low_risk:
            # 对比两组的最佳恢复效果
            high_best = [max(pl["target_logprob"] for pl in r["patching_per_layer"])
                         - r["corrupted_target_logprob"] for r in high_risk]
            low_best = [max(pl["target_logprob"] for pl in r["patching_per_layer"])
                        - r["corrupted_target_logprob"] for r in low_risk]
            print(f"  High-risk 最佳恢复: mean={np.mean(high_best):.2f}")
            print(f"  Low-risk 最佳恢复:  mean={np.mean(low_best):.2f}")
            if np.mean(high_best) > np.mean(low_best) + 0.5:
                print("  ✓ High-risk 组干预效果显著更大 → '信号丢失'是因")
            else:
                print("  ? 两组差异不大 → 需要更多样本")

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
