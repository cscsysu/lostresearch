"""
第六章: InfoDyn-Bench 构建

把所有实验数据整理成标准 Benchmark 发布包:
- Tier 1: 派生轨迹 (CIS, rank, labels) — 轻量, 社区主用
- Tier 2: 完整结果 (含 top5, 生成答案) — 中等
- Tier 3: 审计集说明 — 指向原始数据

输出:
- infodyn_bench_v1.json (主数据)
- infodyn_bench_labels.json (预测标签)
- infodyn_bench_metadata.json (元数据)
- BENCHMARK_README.md (使用说明)
"""
import json
import os
import numpy as np

import config


def build_tier1_trajectories(all_results):
    """Tier 1: 派生轨迹 (轻量, 社区主用)."""
    tier1 = []
    for s in all_results:
        entry = {
            "id": s["id"],
            "task": s.get("task", "unknown"),
            "question": s["question"],
            "answer": s["answer"],
            "aliases": s.get("aliases", [s["answer"]]),
            "generated": s.get("generated", ""),
            "final_correct": s["final_correct"],
            "model": "Qwen3-8B",
            "num_layers": s["num_layers"],
            "trajectory": {
                "cis": s["cis"],
                "correct_rank": s["correct_rank"],
            },
            "derived_metrics": compute_derived_metrics(s),
            "prediction_labels": compute_prediction_labels(s),
        }
        tier1.append(entry)
    return tier1


def compute_derived_metrics(s):
    """计算派生指标."""
    cis = s.get("cis", [])
    ranks = s.get("correct_rank", [])
    n = len(cis)
    if n < 4:
        return {}

    # SEL: 首次 rank <= 5 的相对深度
    sel = 1.0
    for i, r in enumerate(ranks):
        if r <= 5:
            sel = i / (n - 1)
            break

    # Peak rank (排除首尾层)
    mid_ranks = ranks[1:-1]
    peak_rank = min(mid_ranks) if mid_ranks else min(ranks)

    # Peak CIS
    mid_cis = cis[1:-1]
    peak_cis = max(mid_cis) if mid_cis else max(cis)

    # Final CIS
    final_cis = cis[-1]

    # Dwell time (CIS > 0 的比例)
    dwell = sum(1 for c in cis if c > 0) / n

    # Peak-to-final decay
    decay = peak_cis - final_cis

    # Sign change
    sign_change = any(cis[i-1] > 0 and cis[i] < 0 for i in range(1, n))

    # Competitive-decay label
    competitive_decay = any(
        mid_cis[i] > 0 and mid_ranks[i] <= 5
        for i in range(len(mid_cis))
    ) and final_cis < 0

    # Pattern classification
    if peak_rank > 50:
        pattern = "Absent"
    elif competitive_decay:
        pattern = "Preservation-Failure"
    elif sel >= 0.65:
        pattern = "Late-Emergent"
    elif peak_cis > 0 and final_cis < 0:
        pattern = "Decay"
    else:
        pattern = "Stable"

    return {
        "SEL": round(sel, 4),
        "peak_rank": peak_rank,
        "peak_cis": round(peak_cis, 4),
        "final_cis": round(final_cis, 4),
        "dwell_time": round(dwell, 4),
        "peak_to_final_decay": round(decay, 4),
        "sign_change": sign_change,
        "competitive_decay": competitive_decay,
        "pattern": pattern,
        "top5_ever": peak_rank <= 4,
    }


def compute_prediction_labels(s):
    """计算预测任务标签 (在 50% 深度处)."""
    cis = s.get("cis", [])
    n = len(cis)
    if n < 4:
        return {}

    t0_idx = max(2, int(n * 0.5))
    cis_at_t0 = cis[t0_idx]
    cis_final = cis[-1]
    cis_max_mid = max(cis[1:-1]) if n > 2 else max(cis)

    return {
        "t0": 0.5,
        "cis_at_t0": round(cis_at_t0, 4),
        "will_decay": int(cis_max_mid > 0 and cis_final < 0),
        "final_correct": int(s["final_correct"]),
        "final_cis": round(cis_final, 4),
    }


def build_cross_model_tier1(model_key):
    """跨模型 Tier 1."""
    f = os.path.join(config.DATA_DIR, f"cross_model_{model_key}.json")
    if not os.path.exists(f):
        return []
    with open(f) as fh:
        data = json.load(fh)
    results = data.get("trajectory_results", [])
    model_name = data.get("model", model_key)

    tier1 = []
    for s in results:
        entry = {
            "id": s["id"],
            "task": s.get("task", "unknown"),
            "question": s["question"],
            "answer": s["answer"],
            "generated": s.get("generated", ""),
            "final_correct": s["final_correct"],
            "model": model_name,
            "num_layers": len(s.get("cis", [])),
            "trajectory": {
                "cis": s.get("cis", []),
                "correct_rank": s.get("correct_rank", []),
            },
            "derived_metrics": compute_derived_metrics(s),
            "prediction_labels": compute_prediction_labels(s),
        }
        tier1.append(entry)
    return tier1


def build_benchmark():
    """构建完整 Benchmark."""
    print("=" * 70)
    print("Building InfoDyn-Bench v1.0")
    print("=" * 70)

    # 1. Qwen3-8B 主数据
    qwen_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(qwen_file):
        print(f"ERROR: {qwen_file} not found")
        return
    with open(qwen_file) as f:
        qwen_results = json.load(f)
    print(f"Qwen3-8B: {len(qwen_results)} samples")

    qwen_tier1 = build_tier1_trajectories(qwen_results)

    # 2. 跨模型数据
    llama_tier1 = build_cross_model_tier1("llama")
    mistral_tier1 = build_cross_model_tier1("mistral")
    print(f"Llama-3.1-8B: {len(llama_tier1)} samples")
    print(f"Mistral-7B: {len(mistral_tier1)} samples")

    # 3. 合并
    all_tier1 = qwen_tier1 + llama_tier1 + mistral_tier1

    # 4. 统计
    models = set(s["model"] for s in all_tier1)
    tasks = set(s["task"] for s in all_tier1)
    correct = sum(1 for s in all_tier1 if s["final_correct"])
    incorrect = len(all_tier1) - correct
    comp_decay = sum(1 for s in all_tier1 if s["derived_metrics"].get("competitive_decay"))

    # 5. 保存主数据
    bench_file = os.path.join(config.OUTPUT_DIR, "infodyn_bench_v1.json")
    with open(bench_file, "w") as f:
        json.dump(all_tier1, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {bench_file}")
    print(f"Total: {len(all_tier1)} samples, {len(models)} models, {len(tasks)} tasks")
    print(f"Correct: {correct}, Incorrect: {incorrect}")
    print(f"Competitive-decay: {comp_decay} ({100*comp_decay/incorrect:.1f}% of incorrect)")

    # 6. 元数据
    metadata = {
        "name": "InfoDyn-Bench v1.0",
        "description": "Layer-wise Information Dynamics trajectories for LLM error analysis",
        "models": list(models),
        "tasks": list(tasks),
        "total_samples": len(all_tier1),
        "correct_samples": correct,
        "incorrect_samples": incorrect,
        "competitive_decay_samples": comp_decay,
        "metrics": ["CIS", "correct_rank", "SEL", "dwell_time", "peak_to_final_decay",
                     "sign_change", "competitive_decay", "pattern"],
        "prediction_labels": ["will_decay", "final_correct", "final_cis", "cis_at_t0"],
        "patterns": ["Absent", "Preservation-Failure", "Late-Emergent", "Decay", "Stable"],
        "num_layers": {"Qwen3-8B": 36, "Llama-3.1-8B": 32, "Mistral-7B-v0.3": 32},
        "tasks_detail": {
            "triviaqa": "Fact QA (entity-level)",
            "hotpotqa": "Multi-hop reasoning",
            "gsm8k": "Math reasoning",
        },
        "key_findings": {
            "endpoint_bias": "88% pairwise CIS sign reversal is selection bias (null=88.5%)",
            "competitive_decay_rate": "8.9% under tuned lens (raw 5.2%)",
            "behavioral_necessity": "Gold direction ablation: 76%/73%/40% flips vs 0% random",
            "prediction_auc": "0.79 (within-model, vs 0.51 single-layer), structure-dependent cross-task (TQA->HQA 0.74, HQA->GSM 0.77, TQA->GSM 0.38), 0.73 (cross-model Qwen->Llama)",
            "mlp_localization": "Layer 35 MLP is largest negative CIS contributor (-4.55 DLA)",
        },
    }
    meta_file = os.path.join(config.OUTPUT_DIR, "infodyn_bench_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_file}")

    # 7. 预测标签文件 (方便直接用)
    pred_labels = []
    for s in all_tier1:
        pl = s["prediction_labels"]
        if pl:
            pred_labels.append({
                "id": s["id"],
                "model": s["model"],
                "task": s["task"],
                **pl,
            })
    pred_file = os.path.join(config.OUTPUT_DIR, "infodyn_bench_labels.json")
    with open(pred_file, "w") as f:
        json.dump(pred_labels, f, indent=2)
    print(f"Labels: {pred_file}")

    # 8. README
    readme = f"""# InfoDyn-Bench v1.0

## Summary
- Total samples: {len(all_tier1)}
- Models: {', '.join(models)}
- Tasks: {', '.join(tasks)}
- Correct: {correct}, Incorrect: {incorrect}
- Competitive-decay: {comp_decay} ({100*comp_decay/incorrect:.1f}% of incorrect)

## Files
- `infodyn_bench_v1.json` — Main data (trajectories + metrics + labels)
- `infodyn_bench_labels.json` — Prediction labels only (for quick loading)
- `infodyn_bench_metadata.json` — Metadata and key findings

## Data Format
Each sample contains:
```json
{{
  "id": "triviaqa_0000",
  "task": "triviaqa",
  "model": "Qwen3-8B",
  "question": "...",
  "answer": "...",
  "generated": "...",
  "final_correct": true,
  "num_layers": 36,
  "trajectory": {{
    "cis": [0.1, 0.2, ...],
    "correct_rank": [100, 50, ...]
  }},
  "derived_metrics": {{
    "SEL": 0.3,
    "peak_rank": 0,
    "peak_cis": 2.5,
    "final_cis": -1.2,
    "dwell_time": 0.4,
    "peak_to_final_decay": 3.7,
    "sign_change": true,
    "competitive_decay": false,
    "pattern": "Stable"
  }},
  "prediction_labels": {{
    "t0": 0.5,
    "cis_at_t0": 1.2,
    "will_decay": 0,
    "final_correct": 1,
    "final_cis": -1.2
  }}
}}
```

## Metrics
- **CIS**: log P(correct answer) - log P(generated answer) per layer
- **SEL**: Signal Emergence Layer (first layer where rank ≤ 5)
- **peak_rank**: Best (lowest) rank across intermediate layers
- **dwell_time**: Fraction of layers with CIS > 0
- **peak_to_final_decay**: Peak CIS - final CIS
- **competitive_decay**: rank ≤ 5 AND CIS > 0 at some layer AND final CIS < 0

## Patterns
- **Absent**: peak_rank > 50 (signal never formed)
- **Preservation-Failure**: competitive_decay = true
- **Late-Emergent**: SEL ≥ 0.65
- **Decay**: peak CIS > 0 but final CIS < 0
- **Stable**: signal maintained

## Key Findings
1. 88% pairwise CIS sign reversal is selection bias (null = 88.5%)
2. 8.9% competitive-decay under tuned lens (raw = 5.2%, tuned > raw)
3. Gold direction ablation: 76%/73%/40% flips vs 0% random token
4. Prediction AUC: 0.79 (within, vs 0.51 single-layer); cross-task structure-dependent (TQA→HQA 0.74, HQA→GSM 0.77, TQA→GSM 0.38); cross-model 0.73 (Qwen→Llama)
5. Layer 35 MLP: largest negative CIS contributor (-4.55)

## Usage
```python
import json
with open("infodyn_bench_v1.json") as f:
    data = json.load(f)
# Filter by model
qwen = [s for s in data if s["model"] == "Qwen3-8B"]
# Filter by pattern
preservation_failures = [s for s in data if s["derived_metrics"]["pattern"] == "Preservation-Failure"]
```
"""
    readme_file = os.path.join(config.OUTPUT_DIR, "BENCHMARK_README.md")
    with open(readme_file, "w") as f:
        f.write(readme)
    print(f"README: {readme_file}")

    print(f"\n{'='*70}")
    print("InfoDyn-Bench v1.0 Build Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    build_benchmark()
