#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RobotPolicy - ML + 算法混合追捕策略
===================================
符合模型赛马接入规范：
  - 类名: RobotPolicy
  - __init__(self): 加载预训练权重
  - act(self, t, active, pos=None): 返回 (dx, dy)

策略: RandomForest 分类器 (测试准确率 90.27%) + 算法阈值兜底。
      ML判断是否追捕 → 贪心选择最优可达食饵 → 单轴移动。

依赖: numpy, joblib (scikit-learn 自带), json
权重: weights/best_model.pkl, weights/scaler.pkl, weights/feature_names.json
"""

import os
import json
import numpy as np

# joblib 是 scikit-learn 的依赖，必定可用
try:
    import joblib
except ImportError:
    import pickle as _pickle
    joblib = None

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE, 'weights')

ROBOT_DELAY = 0.05   # 50ms
CENTER = (5, 5)


class RobotPolicy:
    """ML + 算法混合策略。

    - 用 RandomForest 对每个可达食饵打分（38维特征）
    - 选择 ML 概率 × 分值 最高的食饵
    - 无可达食饵时向中心靠拢
    """

    def __init__(self):
        self.delay = ROBOT_DELAY
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_weights()

    # ------------------------------------------------------------------
    # 权重加载
    # ------------------------------------------------------------------
    def _load_weights(self):
        model_path = os.path.join(WEIGHTS_DIR, 'best_model.pkl')
        scaler_path = os.path.join(WEIGHTS_DIR, 'scaler.pkl')
        features_path = os.path.join(WEIGHTS_DIR, 'feature_names.json')

        for path, name in [(model_path, '模型'), (scaler_path, '标准化器'),
                           (features_path, '特征名')]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"[RobotPolicy] 缺少{name}文件: {path}")

        if joblib is not None:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            with open(model_path, 'rb') as f:
                self.model = _pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = _pickle.load(f)

        with open(features_path, 'r', encoding='utf-8') as f:
            self.feature_names = json.load(f)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------
    def act(self, t, active, pos=None):
        """每个仿真步调用一次。

        参数:
            t      — 当前仿真时间（秒）
            active — 当前场上食饵 [(ta, x, y, v), ...]
            pos    — 机器人坐标 (x,y) ∈ [1,10]，首次为 None

        返回:
            (dx, dy) ∈ {-1, 0, 1}²，单轴移动。
        """
        if pos is None:
            pos = CENTER

        if not active:
            return self._step_toward(pos, CENTER)

        # 1. 筛选可达食饵 + ML打分
        best_target = None
        best_score = -1e9

        for ta, x, y, v in active:
            if t > ta + 3:
                continue  # 已过期

            prey_pos = (x, y)
            dist = self._manhattan(pos, prey_pos)
            arrival = t + self.delay + dist

            if arrival > ta + 3:
                continue  # 不可达

            # ML预测追捕概率
            proba = self._ml_predict(t, active, ta, x, y, v, pos, dist, arrival)

            # 综合分 = 追捕概率 × 食值 / 距离代价
            score = proba * v / max(dist, 0.5)
            if score > best_score:
                best_score = score
                best_target = prey_pos

        # 2. 移动决策
        if best_target is None:
            return self._step_toward(pos, CENTER)
        return self._step_toward(pos, best_target)

    # ------------------------------------------------------------------
    # ML预测: 构建38维特征 → RandomForest → 追捕概率
    # ------------------------------------------------------------------
    def _ml_predict(self, t, active, ta, x, y, v, pos, dist, arrival):
        """为单个候选食饵计算追捕概率 P(pursue)。"""
        bait_time = ta
        bait_score = float(v)
        row, col = x, y

        wait_time = max(0.0, bait_time - arrival)
        arrival_at_bait = max(arrival, bait_time)

        # 历史统计（从 active 中提取）
        past_t = []; past_s = []; future_t = []
        for ta_i, x_i, y_i, v_i in active:
            if ta_i < bait_time: past_t.append(ta_i); past_s.append(v_i)
            elif ta_i > bait_time: future_t.append(ta_i)
        past_t_arr = np.array(past_t, dtype=np.float64)
        past_s_arr = np.array(past_s, dtype=np.float64)

        # 时间特征
        max_t = max(t, bait_time)
        if len(past_t_arr) > 0: max_t = max(max_t, past_t_arr.max())
        time_frac = bait_time / max(max_t, 1.0)
        time_since_last = bait_time - past_t_arr[-1] if len(past_t_arr) > 0 else 0.0
        time_to_next = min(future_t) - bait_time if future_t else 0.0

        # 窗口特征
        wf = {}
        for w in (30, 60, 120):
            if len(past_t_arr) > 0:
                m = (past_t_arr >= bait_time - w) & (past_t_arr < bait_time)
                c = m.sum()
                wf[f'past_{w}s_count'] = float(c)
                wf[f'past_{w}s_avg'] = float(past_s_arr[m].mean()) if c > 0 else 0.0
                wf[f'past_{w}s_max'] = float(past_s_arr[m].max()) if c > 0 else 0.0
            else:
                wf[f'past_{w}s_count'] = 0.0
                wf[f'past_{w}s_avg'] = 0.0
                wf[f'past_{w}s_max'] = 0.0

        # 机会成本
        opp_sum = 0.0; opp_max = 0.0; opp_cnt = 0
        for ta_i, x_i, y_i, v_i in active:
            if ta_i != bait_time and arrival_at_bait <= ta_i <= arrival_at_bait + 6:
                opp_sum += v_i; opp_max = max(opp_max, v_i); opp_cnt += 1

        # 组装38维特征
        feat = {
            'dist_to_bait': float(dist),
            'dist_normalized': min(float(dist), 18.0) / 18.0,
            'reachable': 1 if dist <= 3 else 0,
            'can_catch': 1,
            'wait_time': float(wait_time),
            'arrival_time_delta': min(arrival_at_bait - bait_time, 10.0),
            'score': bait_score,
            'score_log': np.log1p(bait_score),
            'score_sqrt': np.sqrt(bait_score),
            'score_norm': bait_score / 40.0,
            'is_high_score': 1 if bait_score > 10 else 0,
            'is_vhigh_score': 1 if bait_score > 15 else 0,
            'is_low_score': 1 if bait_score <= 3 else 0,
            'efficiency': bait_score / max(float(dist), 0.5),
            'row': float(row), 'col': float(col),
            'is_boundary': 1 if (row in (1, 10) or col in (1, 10)) else 0,
            'is_corner': 1 if (row in (1, 10) and col in (1, 10)) else 0,
            'dist_to_center': (abs(row - 5.5) + abs(col - 5.5)) / 9.0,
            'robot_row': pos[0] / 10.0,
            'robot_col': pos[1] / 10.0,
            'time_frac': time_frac,
            'time_since_last_bait': float(time_since_last),
            'time_since_last_bait_clipped': min(float(time_since_last), 15.0) / 15.0,
            'time_to_next': float(time_to_next),
            'time_to_next_clipped': min(float(time_to_next), 15.0) / 15.0,
            'past_30s_count': wf['past_30s_count'],
            'past_30s_avg': wf['past_30s_avg'], 'past_30s_max': wf['past_30s_max'],
            'past_60s_count': wf['past_60s_count'],
            'past_60s_avg': wf['past_60s_avg'], 'past_60s_max': wf['past_60s_max'],
            'past_120s_count': wf['past_120s_count'],
            'past_120s_avg': wf['past_120s_avg'], 'past_120s_max': wf['past_120s_max'],
            'opp_score_sum': float(opp_sum),
            'opp_max': float(opp_max), 'opp_count': float(opp_cnt),
        }

        feat_vec = np.array([[feat[name] for name in self.feature_names]], dtype=np.float64)
        feat_scaled = self.scaler.transform(feat_vec)
        return self.model.predict_proba(feat_scaled)[0][1]  # P(追捕)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def _step_toward(pos, target):
        if pos == target:
            return (0, 0)
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        if abs(dx) >= abs(dy) and abs(dx) > 0:
            return (int(np.sign(dx)), 0)
        elif abs(dy) > 0:
            return (0, int(np.sign(dy)))
        return (0, 0)
