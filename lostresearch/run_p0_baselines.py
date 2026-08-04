"""
P0 实验 3+5: 强基线 + 跨任务 predictor transfer

基线:
- Random
- Persistence (current CIS)
- Current CIS only
- Current rank only
- Current entropy (用 top5 近似)
- Max-so-far (CIS peak)
- Local slope
- Answer token frequency (proxy: 用 rank 替代)
- Question length
- Linear
- MLP

跨任务:
- train TriviaQA, test HotpotQA
- train TriviaQA, test GSM8K
- train HotpotQA, test TriviaQA
- leave-one-task-out
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import config
from prediction import extract_features, extract_targets


def compute_entropy_from_top5(top5):
    """从 top5 概率近似熵."""
    if not top5:
        return 0
    probs = np.array([p for _, p in top5[-1]])  # 取最后一层
    probs = probs / probs.sum()
    return -np.sum(probs * np.log(probs + 1e-10))


def extract_strong_features(sample, t0=0.5):
    """提取强基线所需的特征."""
    f = extract_features(sample, t0)
    if f is None:
        return None
    # 加额外特征
    f["question_length"] = len(sample.get("question", ""))
    f["entropy_at_t0"] = compute_entropy_from_top5(sample.get("top5", []))
    return f


def run_strong_baselines(all_samples):
    """强基线实验: 用单个/少量特征预测 decay."""
    print("\n" + "=" * 70)
    print("P0-3: Strong Baselines")
    print("=" * 70)

    features_list, targets_list = [], []
    for s in all_samples:
        f = extract_strong_features(s, config.PREDICTION_T0)
        t = extract_targets(s)
        if f and t:
            features_list.append(f)
            targets_list.append(t)

    if len(features_list) < 20:
        print(f"  Not enough samples: {len(features_list)}")
        return {}

    feature_names = list(features_list[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in features_list])
    y_decay = np.array([t["will_decay"] for t in targets_list])
    y_correct = np.array([t["final_correct"] for t in targets_list])

    results = {}
    feature_idx = {name: i for i, name in enumerate(feature_names)}

    # 单特征基线
    single_features = ["cis_at_t0", "cis_max_before", "cis_slope",
                        "rank_at_t0", "rank_min_before", "logprob_at_t0",
                        "logprob_max_before", "entropy_at_t0", "question_length"]
    print(f"\n  {'Baseline':<25} {'Decay AUC':<12} {'Correct AUC':<14}")
    print("  " + "-" * 51)

    for name in single_features:
        if name not in feature_idx:
            continue
        idx = feature_idx[name]
        x = X[:, idx].reshape(-1, 1)
        if y_decay.sum() > 0 and y_decay.sum() < len(y_decay):
            # 对 decay, 信号越低越可能 decay → 用负值
            auc_decay = roc_auc_score(y_decay, -X[:, idx])
        else:
            auc_decay = 0.5
        if y_correct.sum() > 0 and y_correct.sum() < len(y_correct):
            auc_correct = roc_auc_score(y_correct, X[:, idx])
        else:
            auc_correct = 0.5
        results[f"single_{name}"] = {"decay_auc": auc_decay, "correct_auc": auc_correct}
        print(f"  {name:<25} {auc_decay:<12.3f} {auc_correct:<14.3f}")

    # 多特征基线
    multi_baselines = {
        "cis_only": ["cis_at_t0"],
        "cis_plus_slope": ["cis_at_t0", "cis_slope"],
        "rank_only": ["rank_at_t0"],
        "rank_plus_cis": ["rank_at_t0", "cis_at_t0"],
        "max_sofar": ["cis_max_before"],
        "max_plus_slope": ["cis_max_before", "cis_slope"],
        "all_trajectory": feature_names,
    }

    print()
    for name, feats in multi_baselines.items():
        idxs = [feature_idx[f] for f in feats if f in feature_idx]
        if not idxs:
            continue
        X_sub = X[:, idxs]

        # Decay
        if y_decay.sum() > 0 and y_decay.sum() < len(y_decay):
            clf = LogisticRegression(max_iter=2000, random_state=42)
            clf.fit(X_sub, y_decay)
            auc_decay = roc_auc_score(y_decay, clf.predict_proba(X_sub)[:, 1])
        else:
            auc_decay = 0.5

        # Correct
        if y_correct.sum() > 0 and y_correct.sum() < len(y_correct):
            clf = LogisticRegression(max_iter=2000, random_state=42)
            clf.fit(X_sub, y_correct)
            auc_correct = roc_auc_score(y_correct, clf.predict_proba(X_sub)[:, 1])
        else:
            auc_correct = 0.5

        results[name] = {"decay_auc": auc_decay, "correct_auc": auc_correct,
                          "n_features": len(idxs)}
        print(f"  {name:<25} {auc_decay:<12.3f} {auc_correct:<14.3f}  ({len(idxs)} feats)")

    # MLP (with scaling)
    if len(X) >= 30:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if y_decay.sum() > 0 and y_decay.sum() < len(y_decay):
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                 random_state=42, early_stopping=True)
            mlp.fit(X_scaled, y_decay)
            auc_decay = roc_auc_score(y_decay, mlp.predict_proba(X_scaled)[:, 1])
        else:
            auc_decay = 0.5
        if y_correct.sum() > 0 and y_correct.sum() < len(y_correct):
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                 random_state=42, early_stopping=True)
            mlp.fit(X_scaled, y_correct)
            auc_correct = roc_auc_score(y_correct, mlp.predict_proba(X_scaled)[:, 1])
        else:
            auc_correct = 0.5
        results["mlp_all"] = {"decay_auc": auc_decay, "correct_auc": auc_correct,
                               "n_features": len(feature_names)}
        print(f"  {'mlp_all':<25} {auc_decay:<12.3f} {auc_correct:<14.3f}  ({len(feature_names)} feats)")

    return results


def run_cross_task_transfer(all_samples):
    """跨任务 transfer: 在一个任务上训练, 在另一个任务上测试."""
    print("\n" + "=" * 70)
    print("P0-5: Cross-Task Predictor Transfer")
    print("=" * 70)

    tasks = set(s["task"] for s in all_samples)
    print(f"  Tasks: {tasks}")

    results = {}

    # 提取每个任务的特征和标签
    task_data = {}
    for task in tasks:
        task_samples = [s for s in all_samples if s["task"] == task]
        feats, targets = [], []
        for s in task_samples:
            f = extract_strong_features(s, config.PREDICTION_T0)
            t = extract_targets(s)
            if f and t:
                feats.append(f)
                targets.append(t)
        if len(feats) < 10:
            continue
        feature_names = list(feats[0].keys())
        X = np.array([[f[k] for k in feature_names] for f in feats])
        y_decay = np.array([t["will_decay"] for t in targets])
        y_correct = np.array([t["final_correct"] for t in targets])
        task_data[task] = {"X": X, "y_decay": y_decay, "y_correct": y_correct,
                            "feature_names": feature_names}
        print(f"  {task}: {len(feats)} samples, decay rate={y_decay.mean():.2f}")

    # Cross-task: train on A, test on B
    print(f"\n  {'Train':<12} {'Test':<12} {'Decay AUC':<12} {'Correct AUC':<14}")
    print("  " + "-" * 50)

    for train_task in task_data:
        for test_task in task_data:
            if train_task == test_task:
                continue
            train = task_data[train_task]
            test = task_data[test_task]
            if train["y_decay"].sum() == 0 or train["y_decay"].sum() == len(train["y_decay"]):
                continue

            # Train
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train["X"])
            X_test = scaler.transform(test["X"])

            # Decay
            clf = LogisticRegression(max_iter=2000, random_state=42)
            clf.fit(X_train, train["y_decay"])
            if test["y_decay"].sum() > 0 and test["y_decay"].sum() < len(test["y_decay"]):
                auc_decay = roc_auc_score(test["y_decay"], clf.predict_proba(X_test)[:, 1])
            else:
                auc_decay = 0.5

            # Correct
            if train["y_correct"].sum() > 0 and train["y_correct"].sum() < len(train["y_correct"]):
                clf = LogisticRegression(max_iter=2000, random_state=42)
                clf.fit(X_train, train["y_correct"])
                if test["y_correct"].sum() > 0 and test["y_correct"].sum() < len(test["y_correct"]):
                    auc_correct = roc_auc_score(test["y_correct"], clf.predict_proba(X_test)[:, 1])
                else:
                    auc_correct = 0.5
            else:
                auc_correct = 0.5

            key = f"{train_task}_to_{test_task}"
            results[key] = {"decay_auc": auc_decay, "correct_auc": auc_correct}
            print(f"  {train_task:<12} {test_task:<12} {auc_decay:<12.3f} {auc_correct:<14.3f}")

    # Leave-one-task-out
    print("\n  Leave-one-task-out:")
    for held_out in task_data:
        train_tasks = [t for t in task_data if t != held_out]
        if not train_tasks:
            continue
        X_train = np.vstack([task_data[t]["X"] for t in train_tasks])
        y_train = np.concatenate([task_data[t]["y_decay"] for t in train_tasks])
        y_train_correct = np.concatenate([task_data[t]["y_correct"] for t in train_tasks])

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(task_data[held_out]["X"])

        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(X_train_s, y_train)
        test_y = task_data[held_out]["y_decay"]
        if test_y.sum() > 0 and test_y.sum() < len(test_y):
            auc = roc_auc_score(test_y, clf.predict_proba(X_test_s)[:, 1])
        else:
            auc = 0.5
        print(f"  Hold {held_out:<12}: decay AUC = {auc:.3f}")
        results[f"LOTO_{held_out}"] = {"decay_auc": auc}

    return results


def main():
    results_file = os.path.join(config.DATA_DIR, "full_results_Qwen3-8B.json")
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found")
        return

    with open(results_file) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} samples")

    baselines = run_strong_baselines(all_results)
    transfer = run_cross_task_transfer(all_results)

    out = {"strong_baselines": baselines, "cross_task_transfer": transfer}
    out_file = os.path.join(config.DATA_DIR, "p0_baselines_transfer_Qwen3-8B.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
