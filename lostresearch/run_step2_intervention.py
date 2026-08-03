"""
Step 2: 因果干预 (第五章)
依赖 Step 1 的结果. 先跑 50 题验证代码, 成功后可扩展.
"""
import json
import os
import torch

import config
from data_loader import load_all_datasets, prepare_samples
from intervention import run_intervention_experiment


def main():
    print("=" * 70)
    print("Step 2: Causal Intervention (Chapter 5)")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, dtype=getattr(torch, config.DTYPE),
        device_map=config.DEVICE,
    )
    model.eval()

    # 加载数据
    print("\n[1/3] Loading data...")
    samples = load_all_datasets()
    prepared = prepare_samples(samples, tokenizer)

    # 加载 Step 1 的结果
    results_file = os.path.join(config.DATA_DIR, f"full_results_{config.MODEL_NAME}.json")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found. Run run_step1_collect.py first.")
        return
    print("\n[2/3] Loading Step 1 results...")
    with open(results_file) as f:
        all_results = json.load(f)
    print(f"  Loaded {len(all_results)} samples")

    # 运行因果干预
    print("\n[3/3] Running intervention...")
    results = run_intervention_experiment(model, tokenizer, prepared, all_results)

    # 保存
    def serialize(obj):
        import numpy as np
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj

    out_file = os.path.join(config.DATA_DIR, f"intervention_{config.MODEL_NAME}.json")
    with open(out_file, "w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"\nSaved: {out_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
