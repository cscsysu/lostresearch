"""
第六章: InfoDyn-Bench Benchmark API

提供标准化的数据加载、评测接口, 让社区能复用.
"""
import json
import os
from typing import List, Dict, Optional
import numpy as np

import config


class InfoDynBench:
    """InfoDyn-Bench 数据加载和评测 API.

    用法:
        bench = InfoDynBench()
        bench.load("outputs/data/full_results_Qwen3-8B.json")
        sample = bench.get_sample(0)
        traj = bench.get_trajectory(0)
        metrics = bench.get_metrics(0)
    """

    def __init__(self):
        self.samples = []
        self.model_name = ""

    def load(self, filepath: str):
        """加载轨迹数据."""
        with open(filepath, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        # 从文件名提取模型名
        basename = os.path.basename(filepath)
        if "Qwen3-8B" in basename:
            self.model_name = "Qwen3-8B"
        elif "Llama" in basename:
            self.model_name = "Llama-3.1-8B"
        else:
            self.model_name = "unknown"
        print(f"Loaded {len(self.samples)} samples ({self.model_name})")
        return self

    def get_sample(self, idx: int) -> Dict:
        return self.samples[idx]

    def get_trajectory(self, idx: int) -> Dict:
        """获取样本的层级轨迹."""
        s = self.samples[idx]
        return {
            "correct_logprob": s["correct_logprob"],
            "correct_rank": s["correct_rank"],
            "generated_logprob": s["generated_logprob"],
            "generated_rank": s["generated_rank"],
            "cis": s["cis"],
            "num_layers": s["num_layers"],
        }

    def get_metrics(self, idx: int) -> Dict:
        """获取派生指标."""
        s = self.samples[idx]
        cis = s["cis"]
        ranks = s["correct_rank"]
        n = len(ranks)
        if n < 2:
            return {}

        # SEL
        sel = 1.0
        for i, r in enumerate(ranks):
            if r <= 5:
                sel = i / (n - 1)
                break

        return {
            "SEL": sel,
            "mid_best_rank": min(ranks[1:-1]) if n > 2 else min(ranks),
            "final_rank": ranks[-1],
            "final_cis": cis[-1] if cis else 0,
            "signal_lost": min(ranks[1:-1]) <= 5 and cis and cis[-1] < 0,
            "final_correct": s["final_correct"],
        }

    def get_prediction_labels(self, idx: int, t0: float = 0.5) -> Dict:
        """获取预测任务的标签."""
        s = self.samples[idx]
        cis = s["cis"]
        n = len(cis)
        if n < 4:
            return {}
        t0_idx = max(2, int(n * t0))
        return {
            "will_decay": int(max(cis[1:-1]) > 0 and cis[-1] < 0),
            "final_correct": int(s["final_correct"]),
            "final_cis": cis[-1],
            "cis_at_t0": cis[t0_idx],
        }

    def filter_by_task(self, task: str) -> List[Dict]:
        """按任务过滤."""
        return [s for s in self.samples if s.get("task") == task]

    def filter_by_correct(self, correct: bool) -> List[Dict]:
        """按正确性过滤."""
        return [s for s in self.samples if s["final_correct"] == correct]

    def summary(self) -> Dict:
        """数据集摘要."""
        total = len(self.samples)
        correct = sum(1 for s in self.samples if s["final_correct"])
        tasks = set(s.get("task", "unknown") for s in self.samples)
        return {
            "model": self.model_name,
            "total_samples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0,
            "tasks": list(tasks),
            "num_layers": self.samples[0]["num_layers"] if self.samples else 0,
        }

    def export_for_release(self, output_path: str):
        """导出为发布格式 (去掉大字段, 只保留派生数据)."""
        release = []
        for s in self.samples:
            entry = {
                "id": s["id"],
                "task": s.get("task", "unknown"),
                "question": s["question"],
                "answer": s["answer"],
                "generated": s["generated"],
                "final_correct": s["final_correct"],
                "model": self.model_name,
                "num_layers": s["num_layers"],
                "trajectory": {
                    "correct_logprob": s["correct_logprob"],
                    "correct_rank": s["correct_rank"],
                    "cis": s["cis"],
                },
                "metrics": self.get_metrics(self.samples.index(s)),
            }
            release.append(entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(release, f, ensure_ascii=False, indent=2)
        print(f"Exported {len(release)} samples to {output_path}")


def create_benchmark_release():
    """创建 Benchmark 发布包."""
    bench = InfoDynBench()

    # 加载所有模型的结果
    data_dir = config.DATA_DIR
    all_data = []

    for fname in os.listdir(data_dir):
        if fname.startswith("full_results_") and fname.endswith(".json"):
            filepath = os.path.join(data_dir, fname)
            bench.load(filepath)
            all_data.extend(bench.samples)

    if not all_data:
        print("No data found. Run experiments first.")
        return

    # 导出
    release_path = os.path.join(config.OUTPUT_DIR, "infodyn_bench_release.json")
    bench.samples = all_data
    bench.export_for_release(release_path)

    # 生成 README
    readme_path = os.path.join(config.OUTPUT_DIR, "BENCHMARK_README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# InfoDyn-Bench

## Summary
- Total samples: {len(all_data)}
- Models: {set(s.get('model', 'Qwen3-8B') for s in all_data)}
- Tasks: {set(s.get('task', 'unknown') for s in all_data)}

## Format
Each sample contains:
- `id`, `task`, `question`, `answer`, `generated`
- `final_correct`: whether the model's answer is correct
- `trajectory`: per-layer CIS, correct_logprob, correct_rank
- `metrics`: SEL, mid_best_rank, final_rank, final_cis, signal_lost

## API
```python
from infodyn_bench import InfoDynBench
bench = InfoDynBench()
bench.load("infodyn_bench_release.json")
sample = bench.get_sample(0)
```
""")
    print(f"Benchmark README: {readme_path}")
