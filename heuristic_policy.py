# -*- coding: utf-8 -*-
"""
任务2：手工设计的启发式拦截策略（用于与 PPO 进行 PK）
====================================
设计依据 = 前期统计结论（output/时间规律独立性/任务2前置_可预测性分析报告.md）：
1. 落点不可预测（26项检验 BH 校正后 0 显著）→ 不做落点预测，空闲时驻守
   "覆盖率最优等待点"：在前80%数据的经验落点分布下，曼哈顿 3m（=存活时间×速度）
   可达概率最大的交点。
2. 分值与位置/间隔相互独立 → 追击决策只依赖当前在场食饵，不参考历史。
3. 到达节奏可预测（约94%在 2–10s）→ 拾取后立即回撤等待点，为下一枚做准备
   （回撤本身就是对节奏规律的利用，无需显式计时）。

追击规则：对在场"可达"食饵（曼哈顿距离 ≤ 剩余存活×速度）做带截止期的排列调度
（≤5 个全排列，价值最大、并列取完成时刻最早），奔向最优方案的第一个目标；
无可达食饵时回等待点。

自检: D:/conda/envs/AI/python.exe heuristic_policy.py --selftest
"""
import numpy as np
from itertools import permutations

from bait_env import BaitGridEnv, load_events, GRID, LIFETIME, SPEED, TRAIN_FRAC, EPS


def optimal_wait_point(train_ev=None):
    """前80%数据经验分布下、3m 曼哈顿可达覆盖概率最大的交点（并列取期望距离最小）。"""
    if train_ev is None:
        ev = load_events()
        train_ev = ev[ev[:, 0] <= ev[-1, 0] * TRAIN_FRAC]
    cells = (train_ev[:, 1] * 10 + train_ev[:, 2]).astype(int)
    p = np.bincount(cells, minlength=100) / len(cells)
    cx, cy = np.arange(100) // 10, np.arange(100) % 10
    best, best_cov, best_ed = None, -1.0, 1e18
    for wx in range(GRID + 1):
        for wy in range(GRID + 1):
            d = np.abs(cx - wx) + np.abs(cy - wy)
            cov = p[d <= LIFETIME * SPEED].sum()
            ed = float((p * d).sum())
            if cov > best_cov + 1e-12 or (abs(cov - best_cov) <= 1e-12 and ed < best_ed):
                best, best_cov, best_ed = np.array([wx, wy], float), float(cov), ed
    return best, best_cov


class HeuristicPolicy:
    """act(env) → 方向动作 (ax, ay)∈[-1,1]²，与 BaitGridEnv 动作空间一致。"""

    def __init__(self, train_ev=None):
        self.wait_point, self.coverage = optimal_wait_point(train_ev)

    def _best_first_target(self, env):
        """带截止期的排列调度：返回最优拾取方案的第一个食饵坐标，无可达食饵返回 None。"""
        cand = []
        for a in env.active:
            d = abs(env.pos[0] - a[1]) + abs(env.pos[1] - a[2])
            if d <= (LIFETIME - (env.t - a[0])) * SPEED + EPS:
                cand.append(a)
        if not cand:
            return None
        cand = sorted(cand, key=lambda a: -a[3])[:5]     # 最多5个，5!=120 可承受
        best_v, best_T, best_first = -1.0, 1e18, None
        for perm in permutations(range(len(cand))):
            pos, t, v, first = env.pos.copy(), env.t, 0.0, None
            for i in perm:
                a = cand[i]
                eta = t + (abs(pos[0] - a[1]) + abs(pos[1] - a[2])) / SPEED
                if eta <= a[0] + LIFETIME + EPS:         # 截止期内可赶到才纳入
                    v += a[3]
                    t = eta
                    pos = np.array([a[1], a[2]])
                    if first is None:
                        first = a
            if first is not None and (v > best_v + 1e-9 or
                                      (abs(v - best_v) <= 1e-9 and t < best_T - 1e-9)):
                best_v, best_T, best_first = v, t, first
        return None if best_first is None else np.array([best_first[1], best_first[2]])

    def act(self, env):
        tgt = self._best_first_target(env)
        if tgt is None:
            tgt = self.wait_point
        direction = np.sign(tgt - env.pos)
        return direction.astype(np.float32)              # 全零 → env 原地不动


# ============================================================
def _selftest():
    print('== heuristic selftest ==')
    wp, cov = optimal_wait_point()
    print(f'最优等待点: ({wp[0]:.0f},{wp[1]:.0f})（env 坐标，表格坐标为 '
          f'({wp[0]+1:.0f},{wp[1]+1:.0f})），3m 覆盖概率 {cov:.3f}')

    # 单食饵直线拾取
    env = BaitGridEnv(split='train', synthetic=False, seed=1)
    env.reset(seed=1)
    env.events = env.events[:0]
    env.next_ev = 0
    env.pos = np.array([4.0, 4.0])
    env.active = [np.array([env.t, 6.0, 5.0, 10.0])]     # 3m 外，刚好可达
    pol = HeuristicPolicy()
    got = 0.0
    for _ in range(14):
        _, r, term, trunc, info = env.step(pol.act(env))
        got += max(r, 0)
        if not env.active:
            break
    assert info['picked'] == 1, f'应拾取1个, got picked={info["picked"]}'
    print('直线拾取通过')

    # 验证段 10 窗快速评估（与 PPO 同协议）
    env = BaitGridEnv(split='val', episode_len=600.0, synthetic=False, seed=42)
    scores = []
    for _ in range(10):
        env.reset()
        done = False
        while not done:
            _, r, term, trunc, info = env.step(pol.act(env))
            done = term or trunc
        scores.append(info['score'])
    s = np.array(scores)
    print(f'验证段10窗: {s.mean():.1f} ± {s.std():.1f}（{s.mean()/10:.2f} 分/分钟）')
    print('== heuristic selftest 通过 ==')


if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    _selftest()
