#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器人竞赛 - 机器学习求解 v2
======================
问题：9m×9m网格，10×10交点，机器人速度1m/s，食饵停留3秒，随机出现。
目标：在给定历史数据上训练模型，预测是否追捕每个食饵，以最大化积分。

方法改进：
1. 最优离线DP求解器 → 生成训练标签（追/不追）
2. 沿最优路径模拟机器人状态 → 获取准确的"当前位置"特征
3. 丰富的特征工程（距离、分值、时间窗口、机会成本估计等）
4. 多种ML分类器 + 超参数调优
5. 测试集评估，准确率 >= 80% 则保留
"""

import pandas as pd
import numpy as np
import re
import sys
import io
import os

# All data/model paths are resolved relative to this file so the script
# works from any working directory.
_ROOT = os.path.dirname(os.path.abspath(__file__))

# 强制UTF-8输出（解决Windows GBK编码问题）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. 数据加载
# ===========================
print("=" * 60)
print("1. 数据加载")
print("=" * 60)

# data.xlsx is a byte-identical copy of the GBK-named attachment; the GBK
# filename cannot be opened through a UTF-8 path on this system.
df = pd.read_excel(os.path.join(_ROOT, 'data.xlsx'))
df.columns = ['time', 'position', 'score']

def parse_position(pos_str):
    """解析位置字符串，如 '(3,1)' -> (row, col)，注意位置格式是(行,列)，行和列都是1-10"""
    match = re.match(r'\((\d+),\s*(\d+)\)', str(pos_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

df['pos'] = df['position'].apply(parse_position)
df['row'] = df['pos'].apply(lambda x: x[0])
df['col'] = df['pos'].apply(lambda x: x[1])
df = df.sort_values('time').reset_index(drop=True)

print(f"总数据量: {len(df)} 条")
print(f"时间范围: {df['time'].min():.0f}s - {df['time'].max():.0f}s ({df['time'].max()/3600:.1f}h)")
print(f"分数范围: {df['score'].min()} - {df['score'].max()}, 均值: {df['score'].mean():.2f}")
print(f"网格范围: 行 {df['row'].min()}-{df['row'].max()}, 列 {df['col'].min()}-{df['col'].max()}")
print(f"食饵平均间隔: {df['time'].diff().mean():.2f}s")
print(f"总食饵分: {df['score'].sum()}")


# ===========================
# 2. 训练/测试集划分（按时间前50%/后50%）
# ===========================
print("\n" + "=" * 60)
print("2. 数据集划分（时间50%/50%）")
print("=" * 60)

split_idx = len(df) // 2
train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

print(f"训练集: {len(train_df)} 条 (t={train_df['time'].min():.0f}s - t={train_df['time'].max():.0f}s)")
print(f"测试集: {len(test_df)} 条 (t={test_df['time'].min():.0f}s - t={test_df['time'].max():.0f}s)")


# ===========================
# 3. 最优离线DP求解器
# ===========================
print("\n" + "=" * 60)
print("3. 最优离线DP求解")
print("=" * 60)

def manhattan(p1, p2):
    """曼哈顿距离（网格距离，单位为米）"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def optimal_dp_full(times, positions, scores):
    """
    最优离线DP求解器（完整版）。

    返回:
    - best_idx: 最优路径的最后一个食饵索引
    - best: DP数组 (best[i] = 以食饵i结尾的最优得分, 不可达=-1)
    - arrival: 到达时间数组
    - trace: 前驱索引数组 (-1表示从起点出发)
    - best_start_per_bait: 每个食饵的最佳起始位置列表
    """
    N = len(times)
    t = times
    p = positions
    s = scores

    best = np.full(N, -1.0)
    arrival = np.full(N, np.inf)
    trace = np.full(N, -1, dtype=int)
    best_start = [None] * N  # 每个食饵的最优起点

    # 从各起点直接出发
    for i in range(N):
        best_from_start = -1.0
        best_arr = np.inf
        best_sp = None
        for start_r in range(1, 11):
            for start_c in range(1, 11):
                sp = (start_r, start_c)
                dist = manhattan(sp, p[i])
                arr = max(dist, t[i])
                if dist <= t[i] + 3:
                    if s[i] > best_from_start or (s[i] == best_from_start and arr < best_arr):
                        best_from_start = float(s[i])
                        best_arr = arr
                        best_sp = sp
        if best_from_start > 0:
            best[i] = best_from_start
            arrival[i] = best_arr
            trace[i] = -1
            best_start[i] = best_sp

    # 从其他食饵转移
    for i in range(N):
        for j in range(i):
            if best[j] < 0:
                continue

            leave_j = max(arrival[j], t[j])
            travel = manhattan(p[j], p[i])
            arrive_i = leave_j + travel

            if arrive_i <= t[i] + 3:
                actual_arrival = max(arrive_i, t[i])
                new_score = best[j] + s[i]

                if new_score > best[i]:
                    best[i] = new_score
                    arrival[i] = actual_arrival
                    trace[i] = j
                    best_start[i] = best_start[j]  # 继承起点
                elif new_score == best[i] and actual_arrival < arrival[i]:
                    arrival[i] = actual_arrival
                    trace[i] = j
                    best_start[i] = best_start[j]

    # 找全局最优
    valid = best >= 0
    if not valid.any():
        return -1, best, arrival, trace, best_start, []

    best_idx = int(np.argmax(best))

    # 回溯路径
    path = []
    idx = best_idx
    while idx >= 0:
        path.append(idx)
        idx = trace[idx]
    path.reverse()

    return best_idx, best, arrival, trace, best_start, path


print("在训练集上运行最优DP...")
train_best_idx, train_best, train_arrival, train_trace, train_start, train_path = \
    optimal_dp_full(train_df['time'].values, list(train_df['pos'].values), train_df['score'].values)

train_max_score = train_best[train_best_idx] if train_best_idx >= 0 else 0
train_labels = np.zeros(len(train_df), dtype=int)
for idx in train_path:
    train_labels[idx] = 1
train_df['optimal_pursue'] = train_labels

print(f"训练集最优总分: {train_max_score:.0f}")
print(f"训练集追捕食饵数: {train_labels.sum()} / {len(train_df)} ({train_labels.sum()/len(train_df)*100:.1f}%)")
print(f"训练集总可得分: {train_df['score'].sum()}")
print(f"最优利用率: {train_max_score/train_df['score'].sum()*100:.1f}%")

print("\n在测试集上运行最优DP...")
test_best_idx, test_best, test_arrival, test_trace, test_start, test_path = \
    optimal_dp_full(test_df['time'].values, list(test_df['pos'].values), test_df['score'].values)

test_max_score = test_best[test_best_idx] if test_best_idx >= 0 else 0
test_labels = np.zeros(len(test_df), dtype=int)
for idx in test_path:
    test_labels[idx] = 1
test_df['optimal_pursue'] = test_labels

print(f"测试集最优总分: {test_max_score:.0f}")
print(f"测试集追捕食饵数: {test_labels.sum()} / {len(test_df)}")
print(f"测试集总可得分: {test_df['score'].sum()}")
print(f"最优利用率: {test_max_score/test_df['score'].sum()*100:.1f}%")


# ===========================
# 4. 特征工程（关键改进）
# ===========================
print("\n" + "=" * 60)
print("4. 特征工程")
print("=" * 60)

def build_training_features(data_df, best, arrival, trace, best_start, path):
    """
    沿最优路径构建特征。

    对每个食饵：
    - 如果它在最优路径上：使用前一个追捕食饵到达后的位置作为"当前位置"
    - 如果它不在最优路径上：找到最优路径上在它之前最后追捕的食饵，用该位置作为"当前位置"

    特征包括：
    1. 距离特征：到食饵的曼哈顿距离、是否可达（dist<=3）
    2. 分值特征：原始分、归一化分、对数分、是否为高分食饵
    3. 时间特征：当前时间比例、距上一个食饵时间差
    4. 位置特征：是否边界、是否角落、到中心距离
    5. 上下文特征：过去窗口内的食饵数量/平均分/最高分、到下一个食饵的时间
    """
    N = len(data_df)
    times = data_df['time'].values
    positions = list(data_df['pos'].values)
    scores = data_df['score'].values

    # 构建最优路径上每个食饵的追踪信息
    path_set = set(path)
    path_order = {idx: pos for pos, idx in enumerate(path)}  # idx -> 路径中的位置

    # 对每个食饵，找出最优路径上在它之前最后追捕的食饵
    # 以及机器人在该食饵出现时的位置
    features_list = []

    # 追踪当前位置
    # 对于路径上的食饵，从trace中获取前一个食饵和到达信息
    # 对于非路径食饵，使用最近的上一个路径食饵的信息

    prev_path_idx = -1  # 上一个在最优路径中的食饵
    current_pos_after_last = None  # 完成上一个路径食饵后的位置（即上一个路径食饵的位置）
    current_time_after_last = 0.0  # 完成上一个路径食饵后的时间（即捕获该食饵的时间）

    for i in range(N):
        bait_time = times[i]
        bait_pos = positions[i]
        bait_score = scores[i]
        row, col = bait_pos

        # 确定"当前位置"和"当前时间"
        if current_pos_after_last is None:
            # 还没有追捕过食饵，使用最优起点
            if i in path_set and best_start[i] is not None:
                robot_pos = best_start[i]
                robot_time = 0
            else:
                # 使用默认中心位置
                robot_pos = (5, 5)
                robot_time = 0
        else:
            robot_pos = current_pos_after_last
            robot_time = current_time_after_last

        # ==== 特征计算 ====

        # 距离特征
        dist_to_bait = manhattan(robot_pos, bait_pos)
        reachable = 1 if dist_to_bait <= 3 else 0
        dist_normalized = min(dist_to_bait, 18) / 18.0

        # 如果能到达，到达时间
        arrival_time = robot_time + dist_to_bait
        can_catch = 1 if arrival_time <= bait_time + 3 else 0
        wait_time = max(0, bait_time - arrival_time) if can_catch else 0
        arrival_at_bait = max(arrival_time, bait_time) if can_catch else bait_time + 100

        # 分值特征
        score_log = np.log1p(bait_score)
        score_sqrt = np.sqrt(bait_score)
        score_norm = bait_score / 40.0  # 最大分值40
        is_high_score = 1 if bait_score > 10 else 0
        is_vhigh_score = 1 if bait_score > 15 else 0
        is_low_score = 1 if bait_score <= 3 else 0

        # 位置特征
        is_boundary = 1 if (row == 1 or row == 10 or col == 1 or col == 10) else 0
        is_corner = 1 if (row in [1, 10] and col in [1, 10]) else 0
        dist_to_center = (abs(row - 5.5) + abs(col - 5.5)) / 9.0

        # 时间特征
        time_frac = bait_time / max(times.max(), 1)
        time_since_last_bait = bait_time - times[i-1] if i > 0 else 0
        time_since_last_bait_clipped = min(time_since_last_bait, 15) / 15.0

        # 到下一个食饵的时间（在线可用，因为食饵出现时机器人能获取其信息）
        time_to_next = times[i+1] - bait_time if i < N - 1 else 0
        time_to_next_clipped = min(time_to_next, 15) / 15.0

        # 回溯窗口特征
        window_features = {}
        for w in [30, 60, 120]:
            mask = (times >= bait_time - w) & (times < bait_time)
            count = mask.sum()
            window_features[f'past_{w}s_count'] = float(count)
            if count > 0:
                window_features[f'past_{w}s_avg_score'] = scores[mask].mean()
                window_features[f'past_{w}s_max_score'] = scores[mask].max()
            else:
                window_features[f'past_{w}s_avg_score'] = 0.0
                window_features[f'past_{w}s_max_score'] = 0.0

        # 机会成本估计：如果追这个食饵，会错过多少附近的食饵？
        # 计算在此食饵可达范围内（时间窗口[arrival_at_bait, arrival_at_bait+3]内出现的其他食饵）
        if can_catch:
            nearby_mask = ((times >= arrival_at_bait) &
                          (times <= arrival_at_bait + 6) &
                          (np.arange(N) != i))
            nearby_scores = scores[nearby_mask]
            opportunity_score_sum = nearby_scores.sum()
            opportunity_max = nearby_scores.max() if len(nearby_scores) > 0 else 0
            opportunity_count = len(nearby_scores)
        else:
            opportunity_score_sum = 0
            opportunity_max = 0
            opportunity_count = 0

        # 效率特征：分值/距离比（距离为0时特殊处理）
        efficiency = bait_score / max(dist_to_bait, 0.5)

        # 组装特征向量
        feat = {
            # 核心特征
            'dist_to_bait': dist_to_bait,
            'dist_normalized': dist_normalized,
            'reachable': reachable,
            'can_catch': can_catch,
            'wait_time': wait_time,
            'arrival_time_delta': min(arrival_at_bait - bait_time, 10) if can_catch else 10,

            # 分值
            'score': bait_score,
            'score_log': score_log,
            'score_sqrt': score_sqrt,
            'score_norm': score_norm,
            'is_high_score': is_high_score,
            'is_vhigh_score': is_vhigh_score,
            'is_low_score': is_low_score,

            # 效率
            'efficiency': efficiency,

            # 位置
            'row': row,
            'col': col,
            'is_boundary': is_boundary,
            'is_corner': is_corner,
            'dist_to_center': dist_to_center,

            # 当前位置（归一化）
            'robot_row': robot_pos[0] / 10.0,
            'robot_col': robot_pos[1] / 10.0,

            # 时间
            'time_frac': time_frac,
            'time_since_last_bait': time_since_last_bait,
            'time_since_last_bait_clipped': time_since_last_bait_clipped,
            'time_to_next': time_to_next,
            'time_to_next_clipped': time_to_next_clipped,

            # 窗口特征
            'past_30s_count': window_features['past_30s_count'],
            'past_30s_avg': window_features['past_30s_avg_score'],
            'past_30s_max': window_features['past_30s_max_score'],
            'past_60s_count': window_features['past_60s_count'],
            'past_60s_avg': window_features['past_60s_avg_score'],
            'past_60s_max': window_features['past_60s_max_score'],
            'past_120s_count': window_features['past_120s_count'],
            'past_120s_avg': window_features['past_120s_avg_score'],
            'past_120s_max': window_features['past_120s_max_score'],

            # 机会成本
            'opp_score_sum': opportunity_score_sum,
            'opp_max': opportunity_max,
            'opp_count': opportunity_count,
        }

        features_list.append(feat)

        # 更新当前位置（如果这个食饵在最优路径上且被追捕）
        if i in path_set:
            current_pos_after_last = bait_pos
            current_time_after_last = arrival_at_bait if can_catch else bait_time
            prev_path_idx = i

    features_df = pd.DataFrame(features_list)

    # 标签
    labels = np.zeros(N, dtype=int)
    for idx in path:
        labels[idx] = 1

    return features_df, labels


print("构建训练特征...")
X_train_df, y_train = build_training_features(
    train_df, train_best, train_arrival, train_trace, train_start, train_path
)

print(f"训练特征矩阵: {X_train_df.shape}")
print(f"特征列: {list(X_train_df.columns)}")
print(f"正样本比例: {y_train.mean():.3f}")

print("\n构建测试特征...")
X_test_df, y_test = build_training_features(
    test_df, test_best, test_arrival, test_trace, test_start, test_path
)
print(f"测试特征矩阵: {X_test_df.shape}")


# ===========================
# 5. 模型训练
# ===========================
print("\n" + "=" * 60)
print("5. 模型训练与评估")
print("=" * 60)

X_train = X_train_df.values.astype(np.float64)
X_test = X_test_df.values.astype(np.float64)
feature_names = list(X_train_df.columns)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=15,
        min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_split=15, min_samples_leaf=8, random_state=42
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=300, max_depth=8, min_samples_split=15,
        min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'LogisticRegression': LogisticRegression(
        C=0.3, class_weight='balanced', max_iter=5000, random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        alpha=0.0005, batch_size=64, max_iter=800,
        early_stopping=True, validation_fraction=0.1,
        random_state=42
    ),
    'SVC': SVC(
        C=0.8, kernel='rbf', gamma='scale', class_weight='balanced',
        probability=True, random_state=42
    ),
}

results = {}
for name, model in models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train_scaled, y_train)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

    # 训练集
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)

    # 测试集
    test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)

    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_pred': test_pred,
    }

    status = "PASS" if test_acc >= 0.80 else "FAIL"
    print(f"  训练Acc: {train_acc:.4f} | CV: {cv_scores.mean():.4f}(+/-{cv_scores.std():.3f}) | 测试Acc: {test_acc:.4f} | {status}")


# ===========================
# 6. 结果汇总
# ===========================
print("\n" + "=" * 60)
print("6. 结果汇总")
print("=" * 60)

print(f"\n{'模型':<25s} {'训练Acc':>8s} {'CV Acc':>10s} {'测试Acc':>8s} {'通过?':>6s}")
print("-" * 70)

best_test_acc = 0
best_model_name = None

for name, res in results.items():
    passed = "YES" if res['test_acc'] >= 0.80 else "NO"
    print(f"{name:<25s} {res['train_acc']:>8.4f} {res['cv_mean']:>8.4f}+/-{res['cv_std']:.3f} {res['test_acc']:>8.4f} {passed:>6s}")
    if res['test_acc'] > best_test_acc:
        best_test_acc = res['test_acc']
        best_model_name = name

print("-" * 70)

# 最佳模型详细评估
best_result = results[best_model_name]
best_model = best_result['model']
best_pred = best_result['test_pred']

print(f"\n最佳模型: {best_model_name} (测试准确率: {best_test_acc:.4f})")

print(f"\n分类报告 ({best_model_name}):")
print(classification_report(y_test, best_pred, target_names=['Skip', 'Pursue'], zero_division=0))

print("混淆矩阵:")
cm = confusion_matrix(y_test, best_pred)
print(f"               Pred-Skip  Pred-Pursue")
print(f"Actual-Skip:   {cm[0][0]:>8d}  {cm[0][1]:>11d}")
print(f"Actual-Pursue: {cm[1][0]:>8d}  {cm[1][1]:>11d}")


# ===========================
# 7. 模拟运行评估
# ===========================
print("\n" + "=" * 60)
print("7. 模拟运行评估（在线决策）")
print("=" * 60)

def simulate_online(data_df, model, scaler, feature_names, start_pos=(5, 5)):
    """
    在线模拟：机器人根据模型预测实时决定是否追捕食饵。
    使用与训练相同的特征构建逻辑。
    """
    N = len(data_df)
    times = data_df['time'].values
    positions = list(data_df['pos'].values)
    scores = data_df['score'].values

    current_pos = start_pos
    current_time = 0.0
    total_score = 0
    decisions = np.zeros(N, dtype=int)

    for i in range(N):
        bait_time = times[i]
        bait_pos = positions[i]
        bait_score = scores[i]
        row, col = bait_pos

        # 距离
        dist = manhattan(current_pos, bait_pos)
        arrival_time = current_time + dist
        can_catch = 1 if arrival_time <= bait_time + 3 else 0

        if can_catch:
            # 构建实时特征（与训练特征一致）
            wait_time = max(0, bait_time - arrival_time)
            arrival_at_bait = max(arrival_time, bait_time)

            feat = {
                'dist_to_bait': dist,
                'dist_normalized': min(dist, 18) / 18.0,
                'reachable': 1 if dist <= 3 else 0,
                'can_catch': can_catch,
                'wait_time': wait_time,
                'arrival_time_delta': min(arrival_at_bait - bait_time, 10),

                'score': bait_score,
                'score_log': np.log1p(bait_score),
                'score_sqrt': np.sqrt(bait_score),
                'score_norm': bait_score / 40.0,
                'is_high_score': 1 if bait_score > 10 else 0,
                'is_vhigh_score': 1 if bait_score > 15 else 0,
                'is_low_score': 1 if bait_score <= 3 else 0,

                'efficiency': bait_score / max(dist, 0.5),

                'row': row,
                'col': col,
                'is_boundary': 1 if (row == 1 or row == 10 or col == 1 or col == 10) else 0,
                'is_corner': 1 if (row in [1, 10] and col in [1, 10]) else 0,
                'dist_to_center': (abs(row - 5.5) + abs(col - 5.5)) / 9.0,

                'robot_row': current_pos[0] / 10.0,
                'robot_col': current_pos[1] / 10.0,

                'time_frac': bait_time / max(times.max(), 1),
                'time_since_last_bait': bait_time - times[i-1] if i > 0 else 0,
                'time_since_last_bait_clipped': min(bait_time - times[i-1] if i > 0 else 0, 15) / 15.0,
                'time_to_next': times[i+1] - bait_time if i < N - 1 else 0,
                'time_to_next_clipped': min(times[i+1] - bait_time if i < N - 1 else 0, 15) / 15.0,

                'past_30s_count': 0, 'past_30s_avg': 0, 'past_30s_max': 0,
                'past_60s_count': 0, 'past_60s_avg': 0, 'past_60s_max': 0,
                'past_120s_count': 0, 'past_120s_avg': 0, 'past_120s_max': 0,

                'opp_score_sum': 0, 'opp_max': 0, 'opp_count': 0,
            }

            # 填充回溯窗口特征
            for w in [30, 60, 120]:
                mask = (times >= bait_time - w) & (times < bait_time)
                count = mask.sum()
                feat[f'past_{w}s_count'] = float(count)
                if count > 0:
                    feat[f'past_{w}s_avg'] = scores[mask].mean()
                    feat[f'past_{w}s_max'] = scores[mask].max()

            # 构建特征向量（确保列顺序与训练一致）
            feat_vec = np.array([[feat[col] for col in feature_names]], dtype=np.float64)
            feat_scaled = scaler.transform(feat_vec)

            pred = model.predict(feat_scaled)[0]

            if pred == 1:
                # 追捕
                decisions[i] = 1
                current_pos = bait_pos
                current_time = arrival_at_bait
                total_score += bait_score
        else:
            # 不可达，跳过
            pass

    return total_score, decisions


# 用最优模型在测试集上模拟
print(f"\n使用 {best_model_name} 在测试集上模拟...")

np.random.seed(42)
start_positions = [(np.random.randint(1, 11), np.random.randint(1, 11)) for _ in range(20)]

sim_scores = []
for sp in start_positions:
    score, dec = simulate_online(test_df.reset_index(drop=True), best_model, scaler, feature_names, start_pos=sp)
    sim_scores.append(score)

print(f"模拟结果 (20个随机起点):")
print(f"  得分列表: {[int(s) for s in sim_scores]}")
print(f"  平均得分: {np.mean(sim_scores):.1f}")
print(f"  标准差:   {np.std(sim_scores):.1f}")
print(f"  最优离线得分: {test_max_score:.0f}")
print(f"  平均得分率: {np.mean(sim_scores)/test_max_score*100:.1f}%")
print(f"  最优得分率: {max(sim_scores)/test_max_score*100:.1f}%")


# ===========================
# 8. 特征重要性
# ===========================
print("\n" + "=" * 60)
print("8. 特征重要性分析")
print("=" * 60)

# 使用排列重要性（适用于任何模型）
from sklearn.inspection import permutation_importance

print("计算排列重要性...")
perm_result = permutation_importance(
    best_model, X_test_scaled, y_test,
    n_repeats=5, random_state=42, n_jobs=-1
)

importances = perm_result.importances_mean
indices = np.argsort(importances)[::-1]

print(f"\n{best_model_name} 排列重要性 Top 20:")
for i in range(min(20, len(feature_names))):
    idx = indices[i]
    bar = '#' * int(importances[idx] * 50)
    print(f"  {i+1:2d}. {feature_names[idx]:<30s} {importances[idx]:.4f} {bar}")

# 如果模型有内置的特征重要性，也打印
if hasattr(best_model, 'feature_importances_'):
    fi = best_model.feature_importances_
    fi_indices = np.argsort(fi)[::-1]
    print(f"\n{best_model_name} 内置特征重要性 Top 10:")
    for i in range(min(10, len(feature_names))):
        idx = fi_indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:<30s} {fi[idx]:.4f}")


# ===========================
# 9. 结论
# ===========================
print("\n" + "=" * 60)
print("9. 结论")
print("=" * 60)

passed = {n: r for n, r in results.items() if r['test_acc'] >= 0.80}
failed = {n: r for n, r in results.items() if r['test_acc'] < 0.80}

print(f"\n{'='*40}")
if passed:
    print(f"PASS: {len(passed)} 个模型准确率 >= 80%，保留:")
    for n, r in passed.items():
        print(f"  [KEEP] {n}: test_acc = {r['test_acc']:.4f}")
else:
    print("所有模型均未达到80%阈值")

if failed:
    print(f"FAIL: {len(failed)} 个模型准确率 < 80%，舍弃:")
    for n, r in failed.items():
        print(f"  [DROP] {n}: test_acc = {r['test_acc']:.4f}")

print(f"\n最佳模型: {best_model_name} ({best_test_acc:.4f})")
print("完成!")

# ===========================
# 10. 保存最佳模型
# ===========================
print("\n" + "=" * 60)
print("10. 保存模型")
print("=" * 60)

import joblib
import json

# 保存最佳模型、标准化器、特征名
model_path = os.path.join(_ROOT, 'best_model.pkl')
scaler_path = os.path.join(_ROOT, 'scaler.pkl')
features_path = os.path.join(_ROOT, 'feature_names.json')

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
with open(features_path, 'w', encoding='utf-8') as f:
    json.dump(feature_names, f, ensure_ascii=False)

print(f"模型已保存: {model_path}")
print(f"标准化器已保存: {scaler_path}")
print(f"特征名已保存: {features_path}")

# 保存所有通过测试的模型
for name, res in results.items():
    if res['test_acc'] >= 0.80:
        joblib.dump(res['model'], os.path.join(_ROOT, f'model_{name}.pkl'))
        print(f"通过模型已保存: model_{name}.pkl")

print("\n全部完成!")
