"""
轻量重算 within-model prediction (P1 修复后).

只读已保存的轨迹数据 (full_results_<model>.json 或 cross_model_<model>.json),
用修复后的 extract_features / extract_targets 重新跑 run_prediction_task.
不加载模型、不重新采集, 几秒出结果.

用法:
  python run_prediction_recompute.py --data full_results_Qwen3-8B.json
  python run_prediction_recompute.py --data full_results_Qwen3-8B.json --out my_pred.json
"""
import argparse
import json
import os
import sys

import config
from prediction import run_prediction_task


def load_trajectories(path):
    """加载轨迹数据, 返回样本列表 (要求含 cis / correct_logprob / final_correct)."""
    if not os.path.exists(path):
        sys.exit(f"  ! 文件不存在: {path}")
    with open(path) as f:
        data = json.load(f)
    # 支持两种格式: 顶层就是样本列表, 或含 trajectory_results 的 dict
    if isinstance(data, dict) and "trajectory_results" in data:
        samples = data["trajectory_results"]
    elif isinstance(data, list):
        samples = data
    else:
        sys.exit(f"  ! 无法识别的数据格式: {path}")

    # 检查字段
    missing = [k for k in ("cis", "correct_logprob", "final_correct")
               if k not in samples[0]]
    if missing:
        sys.exit(f"  ! 数据缺字段 {missing}. 需用新版脚本重新采集 "
                 f"(cross_model 需 rerun run_cross_model.py).")
    print(f"  加载 {len(samples)} 条轨迹: {path}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="轨迹数据 json 路径")
    parser.add_argument("--out", default=None, help="输出 json 路径 (默认 <data> 的同名 _recomputed)")
    args = parser.parse_args()

    samples = load_trajectories(args.data)
    results = run_prediction_task(samples)

    out = args.out or args.data.replace(".json", "_recomputed.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  已保存: {out}")


if __name__ == "__main__":
    main()
