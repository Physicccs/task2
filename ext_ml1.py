# -*- coding: utf-8 -*-
"""
同学模型「机器学习1」的统一接口适配器
====================================
原模型：sklearn MLP 二分类，每个新食饵出现时决策 追/跳（inference.py），
追捕时假设机器人沿曼哈顿路径直奔食饵（耗时=距离）。

适配方式：维护与原推理相同的"虚拟状态机"（vpos/vtime 按原 teleport 逻辑推进，
生成追捕目标队列），实际机器人每秒沿队列目标走一格（先 x 后 y）。
曼哈顿耗时与实际走格时间一致，故实际轨迹能精确复现原模型的计划。

在线化差异：原脚本用了 time_to_next（下一食饵出现时刻，未来信息），
在线赛马中不可知，置 0（build_features 对缺失键默认 0 同款处理）。

用法: D:/conda/envs/AI/python.exe task2/pk_task2.py --policy task2/ext_ml1.py
"""
import os
import sys
import json
import numpy as np

MDIR = os.path.join(os.path.dirname(__file__), '..', '算法库', '机器学习1', '机器学习1')
sys.path.insert(0, MDIR)
from inference import build_features          # 复用同学的38维特征构建


class RobotPolicy:
    def __init__(self, start=(5, 5)):
        import joblib
        self.model = joblib.load(os.path.join(MDIR, 'best_model.pkl'))
        self.scaler = joblib.load(os.path.join(MDIR, 'scaler.pkl'))
        with open(os.path.join(MDIR, 'feature_names.json'), encoding='utf-8') as f:
            self.feature_names = json.load(f)
        self.pos = tuple(start)
        self.vpos, self.vtime = tuple(start), 0.0   # 原推理逻辑的虚拟状态
        self.seen = set()
        self.times, self.scores = [], []
        self.plan = []                              # 追捕目标队列 [(x,y),...]

    def _decide(self, b):
        """对一个新出现的食饵按原 inference 逻辑决策。b=(ta,x,y,v)。"""
        ta, x, y, v = b
        self.times.append(float(ta)); self.scores.append(float(v))
        i = len(self.times) - 1
        dist = abs(self.vpos[0] - x) + abs(self.vpos[1] - y)
        arrival = self.vtime + dist
        if arrival > ta + 3.0:                      # 赶不上，跳过
            return
        feats = build_features(self.vpos, self.vtime, ta, (x, y), v,
                               i, np.array(self.times), np.array(self.scores),
                               len(self.times))
        vec = np.array([[feats.get(c, 0.0) for c in self.feature_names]])
        if self.model.predict(self.scaler.transform(vec))[0] == 1:
            self.plan.append((x, y))
            self.vpos = (x, y)
            self.vtime = max(arrival, ta)

    def act(self, t, active, pos=None):
        if pos is not None:
            self.pos = tuple(pos)
        for b in sorted(active, key=lambda b: b[0]):
            key = tuple(b)
            if key not in self.seen:
                self.seen.add(key)
                self._decide(b)
        while self.plan and self.pos == self.plan[0]:
            self.plan.pop(0)
        if not self.plan:
            return (0, 0)
        tx, ty = self.plan[0]
        if self.pos[0] != tx:
            return (1 if tx > self.pos[0] else -1, 0)
        if self.pos[1] != ty:
            return (0, 1 if ty > self.pos[1] else -1)
        return (0, 0)
