"""
Prediction Task: 用中间层信息预测未来信号演化.

在 t0=0.5 (第 50% 深度) 处, 仅用 t<=t0 的信息, 预测:
1. 未来 CIS 是否会衰减 (二分类)
2. 最终是否答对 (二分类)
3. 最终层 CIS 值 (回归)

基线: Random, Persistence, Current CIS, Linear, MLP
"""
import json
import os
from typing import List, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def extract_features(sample: Dict, t0: float = 0.5) -> Dict:
    """提取 t<=t0 的轨迹特征."""
    cis = sample.get("cis", [])
    logprobs = sample["correct_logprob"]
    ranks = sample["correct_rank"]
    n = len(cis)
    if n < 4:
        return None

    t0_idx = max(2, int(n * t0))
    cis_before = cis[:t0_idx]
    lp_before = logprobs[:t0_idx]
    rk_before = ranks[:t0_idx]

    # 特征
    features = {
        "cis_at_t0": cis_before[-1],
        "cis_max_before": max(cis_before),
        "cis_min_before": min(cis_before),
        "cis_mean_before": np.mean(cis_before),
        "cis_slope": (cis_before[-1] - cis_before[0]) / max(len(cis_before), 1),
        "logprob_at_t0": lp_before[-1],
        "logprob_max_before": max(lp_before),
        "logprob_slope": (lp_before[-1] - lp_before[0]) / max(len(lp_before), 1),
        "rank_at_t0": rk_before[-1],
        "rank_min_before": min(rk_before),
        "transitions": sum(1 for i in range(1, len(cis_before))
                          if (cis_before[i] > 0) != (cis_before[i-1] > 0)),
        "cis_variance": np.var(cis_before),
    }
    return features


def extract_targets(sample: Dict) -> Dict:
    """提取预测目标."""
    cis = sample.get("cis", [])
    if len(cis) < 2:
        return None

    t0_idx = max(2, int(len(cis) * config.PREDICTION_T0))
    cis_after_t0 = cis[t0_idx:]
    cis_final = cis[-1]
    cis_max_mid = max(cis[1:-1]) if len(cis) > 2 else max(cis)

    # 是否衰减: 中间高, 最终低
    will_decay = 1 if (cis_max_mid > 0 and cis_final < 0) else 0
    # 是否最终 CIS < 0 (错误信号压过)
    final_negative = 1 if cis_final < 0 else 0
    # 是否答对
    final_correct = 1 if sample["final_correct"] else 0
    # 最终 CIS 值 (回归)
    final_cis_value = cis_final

    return {
        "will_decay": will_decay,
        "final_negative": final_negative,
        "final_correct": final_correct,
        "final_cis_value": final_cis_value,
    }


def run_prediction_task(all_samples: List[Dict]) -> Dict:
    """运行预测任务, 返回各基线的表现."""
    print("\n" + "=" * 70)
    print("PREDICTION TASK")
    print("=" * 70)

    # 提取特征和目标
    features_list = []
    targets_list = []
    for s in all_samples:
        f = extract_features(s, config.PREDICTION_T0)
        t = extract_targets(s)
        if f is not None and t is not None:
            features_list.append(f)
            targets_list.append(t)

    if len(features_list) < 20:
        print(f"  Not enough samples for prediction (n={len(features_list)}), need >= 20")
        return {}

    feature_names = list(features_list[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in features_list])
    # Scale features for MLP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_decay = np.array([t["will_decay"] for t in targets_list])
    y_correct = np.array([t["final_correct"] for t in targets_list])
    y_cis = np.array([t["final_cis_value"] for t in targets_list])

    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Target 'will_decay': {y_decay.sum()}/{len(y_decay)} positive")
    print(f"  Target 'final_correct': {y_correct.sum()}/{len(y_correct)} positive")

    results = {}

    # === 基线 1: Random ===
    np.random.seed(42)
    y_pred_random = np.random.rand(len(y_decay))
    results["random"] = {
        "will_decay_auc": 0.5,
        "final_correct_auc": 0.5,
        "final_cis_mae": float(np.mean(np.abs(y_cis - np.mean(y_cis)))),
    }

    # === 基线 2: Persistence (用当前 CIS 直接预测) ===
    cis_at_t0 = X[:, feature_names.index("cis_at_t0")]
    results["persistence"] = {
        "will_decay_auc": roc_auc_score(y_decay, -cis_at_t0) if y_decay.sum() > 0 else 0.5,
        "final_correct_auc": roc_auc_score(y_correct, cis_at_t0) if y_correct.sum() > 0 else 0.5,
        "final_cis_mae": float(np.mean(np.abs(y_cis - cis_at_t0))),
    }

    # === 基线 3: Current CIS + slope ===
    cis_slope = X[:, feature_names.index("cis_slope")]
    persistence_plus = cis_at_t0 + 2 * cis_slope  # 简单线性外推
    results["persistence_plus"] = {
        "will_decay_auc": roc_auc_score(y_decay, -persistence_plus) if y_decay.sum() > 0 else 0.5,
        "final_correct_auc": roc_auc_score(y_correct, persistence_plus) if y_correct.sum() > 0 else 0.5,
        "final_cis_mae": float(np.mean(np.abs(y_cis - persistence_plus))),
    }

    # === 基线 4: Linear Regression / Logistic ===
    if len(X) >= 20:
        # 回归
        lr = LinearRegression()
        lr.fit(X, y_cis)
        y_pred_cis_lr = lr.predict(X)
        # 分类
        clf_decay = LogisticRegression(max_iter=1000, random_state=42)
        clf_correct = LogisticRegression(max_iter=1000, random_state=42)
        if y_decay.sum() > 0 and y_decay.sum() < len(y_decay):
            clf_decay.fit(X, y_decay)
            y_pred_decay_lr = clf_decay.predict_proba(X)[:, 1]
            auc_decay_lr = roc_auc_score(y_decay, y_pred_decay_lr)
        else:
            auc_decay_lr = 0.5
        if y_correct.sum() > 0 and y_correct.sum() < len(y_correct):
            clf_correct.fit(X, y_correct)
            y_pred_correct_lr = clf_correct.predict_proba(X)[:, 1]
            auc_correct_lr = roc_auc_score(y_correct, y_pred_correct_lr)
        else:
            auc_correct_lr = 0.5
        results["linear"] = {
            "will_decay_auc": auc_decay_lr,
            "final_correct_auc": auc_correct_lr,
            "final_cis_mae": float(np.mean(np.abs(y_cis - y_pred_cis_lr))),
        }

    # === 基线 5: MLP (with scaling) ===
    if len(X) >= 30:
        # 回归
        mlp_reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000,
                               random_state=42, early_stopping=True)
        mlp_reg.fit(X_scaled, y_cis)
        y_pred_cis_mlp = mlp_reg.predict(X_scaled)
        # 分类
        if y_decay.sum() > 0 and y_decay.sum() < len(y_decay):
            mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                     random_state=42, early_stopping=True)
            mlp_clf.fit(X_scaled, y_decay)
            y_pred_decay_mlp = mlp_clf.predict_proba(X_scaled)[:, 1]
            auc_decay_mlp = roc_auc_score(y_decay, y_pred_decay_mlp)
        else:
            auc_decay_mlp = 0.5
        if y_correct.sum() > 0 and y_correct.sum() < len(y_correct):
            mlp_clf2 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                      random_state=42, early_stopping=True)
            mlp_clf2.fit(X_scaled, y_correct)
            y_pred_correct_mlp = mlp_clf2.predict_proba(X_scaled)[:, 1]
            auc_correct_mlp = roc_auc_score(y_correct, y_pred_correct_mlp)
        else:
            auc_correct_mlp = 0.5
        results["mlp"] = {
            "will_decay_auc": auc_decay_mlp,
            "final_correct_auc": auc_correct_mlp,
            "final_cis_mae": float(np.mean(np.abs(y_cis - y_pred_cis_mlp))),
        }

    # === 打印结果 ===
    print(f"\n  {'Baseline':<20} {'Decay AUC':<12} {'Correct AUC':<14} {'CIS MAE':<10}")
    print("  " + "-" * 56)
    for name, r in results.items():
        print(f"  {name:<20} {r['will_decay_auc']:<12.3f} "
              f"{r['final_correct_auc']:<14.3f} {r['final_cis_mae']:<10.3f}")

    print("\n  关键判据:")
    if "persistence" in results and "mlp" in results:
        diff = results["mlp"]["will_decay_auc"] - results["persistence"]["will_decay_auc"]
        if diff > 0.05:
            print(f"  ✓ MLP 比 Persistence 提升 {diff:.3f} AUC → 轨迹历史有额外预测力")
        else:
            print(f"  ✗ MLP 比 Persistence 仅提升 {diff:.3f} AUC → 轨迹历史无额外贡献")

    return results
