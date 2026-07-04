# -*- coding: utf-8 -*-
"""
机制2：追击取舍与调度目标改造（终点感知调度 + 追击门槛 skip 规则 + 候选集扩容）
================================================================================
【最优参数与成绩】
  make_act(mu=0.0, theta=0.0, topk=5)   ← 最优即基线行为，机制判定 reject
  seg A: 33.99 分/分钟, picked 233（基线 33.99）
  seg B: 34.24 分/分钟, picked 102（基线 34.24，锁参后只验证一次）

【sweep 结论（seg A）】
  μ∈{0,0.1,0.2,0.4,0.8,1.2}（θ=0）: 全部 33.99，与基线逐分不差 —— μ 任何取值
    都未改变过任何一步的首目标选择（实测 0 分歧步）。
  θ∈{0,0.3,0.6,0.9,1.2,1.5}（μ=0）: 33.99→33.69→33.04→32.00→30.22→29.23，
    单调受损：砍掉的每次追击都是纯损失。
  组合 3×3 邻域: 最优仍 (0,0)。topk 5→7: Δ=+0.00。
【为什么无效（机制层面）】
  食饵间隔均值≈6.0s、存活 3s → 场上同时在场≈0.5 枚；再经可达过滤后，
  seg A 全程 14304 步中"可达候选数"分布：0 枚 12124 步 / 1 枚 2176 步 /
  2 枚仅 4 步 / ≥3 枚 0 步。排列调度几乎永远只有 0~1 个候选，
  "终点感知打分/扩大候选集"没有决策空间可改写；"追击门槛"只能砍掉
  唯一候选，而追击的机会成本本来就≈0（阶段1诊断：chase 态错失值仅占
  2.3%，return 态拾取率 47.8% 反而是三态最高——离家在中场反离新饵更近）。
  与诊断结论一致：改进空间在暴露度/站位，不在调度取舍。

设计（基于 HeuristicPolicy 复制改造，不修改任何现有文件）：
1. 终点感知调度：排列打分从 (总价值最大, 并列完成最早) 改为
     score = V − μ·T_off,  T_off = (T_完成 − t) + d(方案终点, 等待点)/SPEED
   μ 的量纲 = 分/秒（离家机会成本）。μ>0 时偏好"顺路且打完离家近"的方案。
2. 追击门槛（skip 规则）：仅当 V >= θ·T_off 的方案才允许执行；
   若全部方案被过滤则留守/回撤等待点。每步重估（act 无记忆，天然满足）。
3. 方案枚举 = 候选（按价值取前 topk）全排列的**所有前缀**（即所有有序子集）：
   μ>0/θ>0 时允许主动放弃尾部低价值远目标；μ=θ=0 时价值单调保证完整方案
   占优，行为与基线 HeuristicPolicy 逐步一致（已验证 A 段同分 33.99）。
4. 候选集扩容 topk=5→7（机制3 快速证实收益）。

数据纪律：只用 tune_harness 暴露的前 893 条（A=前600 调参, B=601..893 验证一次）。
运行 sweep： D:/conda/envs/AI/python.exe task2/variants/m2_chase.py
"""
import os
import sys
from itertools import permutations

sys.path.insert(0, 'task2')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np  # noqa: E402
from bait_env import LIFETIME, SPEED, EPS  # noqa: E402
from heuristic_policy import optimal_wait_point  # noqa: E402


class ChasePolicy:
    """act(env) → (ax,ay)∈[-1,1]²。调度目标含离家机会成本 μ，追击门槛 θ。"""

    def __init__(self, mu=0.0, theta=0.0, topk=5, train_ev=None):
        self.mu = float(mu)
        self.theta = float(theta)
        self.topk = int(topk)
        self.wait_point, self.coverage = optimal_wait_point(train_ev)

    def _best_first_target(self, env):
        """带截止期的有序子集调度：返回通过 θ 门槛、μ-score 最高方案的首目标。"""
        cand = []
        for a in env.active:
            d = abs(env.pos[0] - a[1]) + abs(env.pos[1] - a[2])
            if d <= (LIFETIME - (env.t - a[0])) * SPEED + EPS:
                cand.append(a)
        if not cand:
            return None
        cand = sorted(cand, key=lambda a: -a[3])[:self.topk]
        wx, wy = self.wait_point[0], self.wait_point[1]
        best_s, best_T, best_first = -1e18, 1e18, None
        for perm in permutations(range(len(cand))):
            px, py, t, v, first = env.pos[0], env.pos[1], env.t, 0.0, None
            for i in perm:
                a = cand[i]
                eta = t + (abs(px - a[1]) + abs(py - a[2])) / SPEED
                if eta <= a[0] + LIFETIME + EPS:      # 截止期内可赶到才纳入
                    v += a[3]
                    t = eta
                    px, py = a[1], a[2]
                    if first is None:
                        first = a
                    # —— 前缀即一个完整候选方案：θ 过滤 + μ-score 择优 ——
                    t_off = (t - env.t) + (abs(px - wx) + abs(py - wy)) / SPEED
                    if v < self.theta * t_off - 1e-9:
                        continue                       # 不值得离家（门槛过滤）
                    s = v - self.mu * t_off
                    if s > best_s + 1e-9 or (abs(s - best_s) <= 1e-9 and t < best_T - 1e-9):
                        best_s, best_T, best_first = s, t, first
        return None if best_first is None else np.array([best_first[1], best_first[2]])

    def act(self, env):
        tgt = self._best_first_target(env)
        if tgt is None:
            tgt = self.wait_point
        return np.sign(tgt - env.pos).astype(np.float32)


def make_act(mu=0.0, theta=0.0, topk=5):
    """交付约定：工厂 → act(env)。默认参数将在 sweep 后更新为最优组合。"""
    pol = ChasePolicy(mu=mu, theta=theta, topk=topk)
    return pol.act


# ==========================================================================
def _sweep():
    from tune_harness import evaluate

    def run(mu, theta, topk=5, seg='A'):
        r = evaluate(lambda: make_act(mu=mu, theta=theta, topk=topk), seg=seg)
        return r['per_min'], r['picked']

    print('== 0) 等价性检查：mu=0, theta=0, topk=5 应复现基线 A=33.99 ==')
    pm, pk = run(0.0, 0.0)
    print(f'   A: {pm:.2f} 分/分钟, picked {pk}')

    print('\n== 1) 单独扫 μ（θ=0） ==')
    mu_grid = [0.0, 0.1, 0.2, 0.4, 0.8, 1.2]
    mu_res = {}
    for mu in mu_grid:
        pm, pk = run(mu, 0.0)
        mu_res[mu] = pm
        print(f'   mu={mu:<4} A={pm:6.2f}  picked={pk}')

    print('\n== 2) 单独扫 θ（μ=0） ==')
    th_grid = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
    th_res = {}
    for th in th_grid:
        pm, pk = run(0.0, th)
        th_res[th] = pm
        print(f'   theta={th:<4} A={pm:6.2f}  picked={pk}')

    best_mu = max(mu_res, key=mu_res.get)
    best_th = max(th_res, key=th_res.get)
    print(f'\n单机制最优: mu*={best_mu} (A={mu_res[best_mu]:.2f}), '
          f'theta*={best_th} (A={th_res[best_th]:.2f})')

    print('\n== 3) 组合扫最优邻域（3×3） ==')
    def neigh(x, grid):
        i = grid.index(x)
        lo = max(i - 1, 0)
        return grid[lo:lo + 3] if lo + 3 <= len(grid) else grid[-3:]

    combo = {}
    for mu in neigh(best_mu, mu_grid):
        for th in neigh(best_th, th_grid):
            pm, pk = run(mu, th)
            combo[(mu, th)] = pm
            print(f'   mu={mu:<4} theta={th:<4} A={pm:6.2f}  picked={pk}')
    best_combo = max(combo, key=combo.get)
    print(f'组合最优: mu={best_combo[0]}, theta={best_combo[1]}, A={combo[best_combo]:.2f}')

    print('\n== 4) 机制3：候选集 top5 → top7 / top6按(价值,-距离) ==')
    for mu, th in {(0.0, 0.0), best_combo}:
        pm5, _ = run(mu, th, topk=5)
        pm7, _ = run(mu, th, topk=7)
        print(f'   (mu={mu},theta={th}) top5 A={pm5:.2f} | top7 A={pm7:.2f} | Δ={pm7-pm5:+.2f}')

    return best_combo, combo[best_combo]


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    best, bestA = _sweep()
    print('\n（seg B 验证由主流程在锁定参数后单独执行一次）')
