"""
跨模型 predictor transfer: 在 Qwen 上训练, 在 Llama/Mistral 上测试.

这证明轨迹动态规律不是模型特异的.
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from prediction import extract_features, extract_targets


def load_model_results(model_key):
    """加载某个模型的结果."""
    if model_key == "qwen":
        f = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    else:
        f = os.path.join(config.DATA_DIR, f"cross_model_{model_key}.json")

    if not os.path.exists(f):
        print(f"  {f} not found")
        return None

    with open(f) as fh:
        data = json.load(fh)

    if model_key == "qwen":
        return data
    else:
        return data.get("trajectory_results", [])


def extract_features_targets(samples, t0=0.5):
    """提取特征和标签.

    要求样本含真实 correct_logprob (gold log-probability 轨迹).
    不再用 cis 近似替换, 否则 train/test 的 logprob 特征物理量不一致,
    transfer 结果无效. 缺 correct_logprob 的旧数据会被跳过并计数,
    需用更新版 run_cross_model.py 重跑生成.
    """
    features_list, targets_list = [], []
    skipped = 0
    for s in samples:
        if "correct_logprob" not in s:
            skipped += 1
            continue
        f = extract_features(s, t0)
        t = extract_targets(s)
        if f and t:
            features_list.append(f)
            targets_list.append(t)
    if skipped:
        print(f"  ! skipped {skipped} samples missing correct_logprob "
              f"(rerun run_cross_model.py to regenerate)")
    return features_list, targets_list


def run_cross_model_prediction():
    """跨模型 predictor transfer."""
    print("\n" + "=" * 70)
    print("Cross-Model Predictor Transfer")
    print("=" * 70)

    models = {"qwen": "Qwen3-8B", "llama": "Llama-3.1-8B", "mistral": "Mistral-7B"}

    # 加载所有模型的数据
    model_data = {}
    for key, name in models.items():
        print(f"\nLoading {name}...")
        data = load_model_results(key)
        if data:
            features, targets = extract_features_targets(data)
            if len(features) >= 10:
                feature_names = list(features[0].keys())
                X = np.array([[f[k] for k in feature_names] for f in features])
                y_decay = np.array([t["will_decay"] for t in targets])
                y_correct = np.array([t["final_correct"] for t in targets])
                model_data[key] = {
                    "name": name,
                    "X": X, "y_decay": y_decay, "y_correct": y_correct,
                    "feature_names": feature_names,
                    "n": len(features),
                    "decay_rate": y_decay.mean(),
                }
                print(f"  {name}: {len(features)} samples, decay rate={y_decay.mean():.2f}")

    if len(model_data) < 2:
        print("Not enough models for cross-model transfer")
        return

    # Cross-model transfer
    print(f"\n  {'Train':<15} {'Test':<15} {'Decay AUC':<12} {'Correct AUC':<14}")
    print("  " + "-" * 56)

    results = {}
    for train_key in model_data:
        for test_key in model_data:
            if train_key == test_key:
                continue

            train = model_data[train_key]
            test = model_data[test_key]

            if train["y_decay"].sum() == 0 or train["y_decay"].sum() == len(train["y_decay"]):
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train["X"])
            X_test = scaler.transform(test["X"])

            # Decay prediction
            clf = LogisticRegression(max_iter=2000, random_state=42)
            clf.fit(X_train, train["y_decay"])

            if test["y_decay"].sum() > 0 and test["y_decay"].sum() < len(test["y_decay"]):
                auc_decay = roc_auc_score(test["y_decay"], clf.predict_proba(X_test)[:, 1])
            else:
                auc_decay = 0.5

            # Correct prediction
            if train["y_correct"].sum() > 0 and train["y_correct"].sum() < len(train["y_correct"]):
                clf2 = LogisticRegression(max_iter=2000, random_state=42)
                clf2.fit(X_train, train["y_correct"])
                if test["y_correct"].sum() > 0 and test["y_correct"].sum() < len(test["y_correct"]):
                    auc_correct = roc_auc_score(test["y_correct"], clf2.predict_proba(X_test)[:, 1])
                else:
                    auc_correct = 0.5
            else:
                auc_correct = 0.5

            key = f"{train_key}_to_{test_key}"
            results[key] = {"decay_auc": auc_decay, "correct_auc": auc_correct}
            print(f"  {train['name']:<15} {test['name']:<15} {auc_decay:<12.3f} {auc_correct:<14.3f}")

    # Leave-one-model-out
    print(f"\n  Leave-one-model-out:")
    for held_out_key in model_data:
        train_keys = [k for k in model_data if k != held_out_key]
        if not train_keys:
            continue

        X_train = np.vstack([model_data[k]["X"] for k in train_keys])
        y_train = np.concatenate([model_data[k]["y_decay"] for k in train_keys])

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(model_data[held_out_key]["X"])

        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_train_s, y_train)

        test_y = model_data[held_out_key]["y_decay"]
        if test_y.sum() > 0 and test_y.sum() < len(test_y):
            auc = roc_auc_score(test_y, clf.predict_proba(X_test_s)[:, 1])
        else:
            auc = 0.5

        print(f"  Hold {model_data[held_out_key]['name']:<15}: decay AUC = {auc:.3f}")
        results[f"LOMO_{held_out_key}"] = {"decay_auc": auc}

    # 判断
    print(f"\n  --- 判断 ---")
    cross_aucs = [v["decay_auc"] for k, v in results.items()
                  if "to" in k and "LOMO" not in k]
    if cross_aucs:
        mean_cross = np.mean(cross_aucs)
        if mean_cross > 0.75:
            print(f"  ✓ 跨模型 transfer 平均 AUC={mean_cross:.3f} > 0.75")
            print(f"  → 轨迹动态规律跨模型通用")
        elif mean_cross > 0.65:
            print(f"  ? 跨模型 transfer 平均 AUC={mean_cross:.3f}, 中等")
        else:
            print(f"  ✗ 跨模型 transfer 平均 AUC={mean_cross:.3f}, 不够强")

    return results


def main():
    results = run_cross_model_prediction()

    out_file = os.path.join(config.DATA_DIR, "cross_model_prediction_transfer.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
