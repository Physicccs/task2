# -*- coding: utf-8 -*-
"""
机制3：路径塑形（追击/回撤路径的顺带覆盖）—— 在最短路集合内选高覆盖走廊
====================================================================
【结论：REJECT（收益不稳健，B 段为负）】A 上最优参数 mode='point'。实测（分/分钟）：
  基线(mode='baseline', 精确复现现行 HeuristicPolicy): A 33.99 / B 34.24（复现通过）
  逐格点覆盖贪心(mode='point'):   A 34.23 (+0.24, 拾233/600) / B 33.97 (-0.27, 拾101/293)
  平滑覆盖走廊贪心(mode='smooth'): A 34.03 (+0.04, 拾232/600) / （A 上劣于 point，未测 B）
  A 段 +0.24 在 B 段翻负 −0.27，效应量 |Δ|≈0.25 且换段变号 → 属分段噪声，非真实增益。
  建议保留基线轴序（先 x 后 y），不采纳本机制。
【为何无效（机制层解释）】
  1) 塑形只在“交点且两轴都剩余”时有选择权，离开等待点的时间本就只占 27.2%，
     实际可塑形的步数更少；且每次选择只让路径横向偏移 ≤1m。
  2) 相邻交点 cov 差仅 0.01~0.05，1m 偏移只对出现瞬间 d∈(2,4] 边缘带的食饵改变可达性，
     该带占错失值比例小，且偏向高覆盖侧必然背离低覆盖侧（收益对冲）。
  3) 等待点 env(2,6) 本身已在高覆盖角，追击/回撤路径天然穿过高覆盖区，
     最短路集合内部的暴露度方差极小。与阶段1诊断一致：损失主因是单点覆盖不足
     （96.4% 错失值出现即不可达 d>3），路径微调无法触及。

【机制】
bait_env._move 在交点处按“方向向量·候选边”点积最大选边；现行策略输出
sign(tgt-pos)=(±1,±1)，点积在 x/y 轴并列，argmax 恒取 x → 走出“先 x 后 y”的固定折线。
曼哈顿最短路不唯一：在 |dx|>0 且 |dy|>0 的交点上，“先走 x 一格”与“先走 y 一格”都朝
目标单调前进、耗时相同。本策略比较这两个相邻交点的覆盖率 cov，选高者，让机器人在
移动途中尽量停留/穿过高覆盖交点，从而对“路上新出现的食饵”暴露度更高（可达概率更大）。
不改变追击调度、不改变等待点、不增加任何耗时（仍是最短路）。

【cov 定义】
- point : cov(c) = P(食饵落点落在 c 的曼哈顿 3m 邻域内)，用 seg_events('A') 经验落点分布估计
          （即“从 c 出发 3s×1m/s 可达的落点概率质量”，与 optimal_wait_point 同口径）。
- smooth: 上述 cov 再在每个交点的 3m 曼哈顿邻域内取均值（走廊值而非点值），
          偏好“整段走廊都高覆盖”的方向，对后续继续前进的方向更鲁棒。

【与基线的唯一差异】只在“对角交点”改变 x/y 轴序的并列裁决；覆盖并列时回退取 x（= 基线）。
mode='baseline' 精确复现现行策略（自测应 ≈ A 33.99）。

数据纪律：cov 只用 seg_events('A')；A 选参、B 仅验证一次。不修改任何现有文件。
运行： D:/conda/envs/AI/python.exe task2/variants/m3_path.py
"""
import sys
import os

sys.path.insert(0, 'task2')
sys.path.insert(0, os.path.join('task2', 'variants'))

import numpy as np  # noqa: E402
from bait_env import GRID, LIFETIME, SPEED, EPS  # noqa: E402
from heuristic_policy import HeuristicPolicy  # noqa: E402
from tune_harness import ReplayEnv  # noqa: E402
from tune_harness import seg_events, evaluate  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

N = GRID + 1        # 交点数/轴 = 10（坐标 0..9）
REACH = LIFETIME * SPEED   # 3m 可达半径


def coverage_point():
    """cov(c)=落点落入 c 的曼哈顿 3m 邻域的经验概率（seg A）。返回 (N,N) 数组，索引 [x,y]。"""
    ev = seg_events('A')
    cells = (ev[:, 1] * 10 + ev[:, 2]).astype(int)
    p = np.bincount(cells, minlength=100) / len(cells)
    cx, cy = np.arange(100) // 10, np.arange(100) % 10
    cov = np.zeros((N, N))
    for wx in range(N):
        for wy in range(N):
            d = np.abs(cx - wx) + np.abs(cy - wy)
            cov[wx, wy] = p[d <= REACH + EPS].sum()
    return cov


def coverage_smooth(cov):
    """在每个交点的 3m 曼哈顿邻域内取 cov 均值（走廊平滑值）。"""
    covs = np.zeros((N, N))
    for wx in range(N):
        for wy in range(N):
            tot, cnt = 0.0, 0
            for ax in range(N):
                for ay in range(N):
                    if abs(ax - wx) + abs(ay - wy) <= REACH + EPS:
                        tot += cov[ax, ay]
                        cnt += 1
            covs[wx, wy] = tot / cnt
    return covs


class PathShapePolicy:
    """追击/等待逻辑完全复用 HeuristicPolicy；仅在对角交点用 cov 决定先走哪条轴。"""

    def __init__(self, mode='smooth', train_ev=None):
        self.mode = mode
        self.h = HeuristicPolicy(train_ev)     # 复用等待点 + 带截止期排列调度
        self.wait_point = self.h.wait_point
        if mode == 'baseline':
            self.C = None
        else:
            cov = coverage_point()
            self.C = cov if mode == 'point' else coverage_smooth(cov)

    def act(self, env):
        tgt = self.h._best_first_target(env)
        if tgt is None:
            tgt = self.wait_point
        pos = env.pos
        dx, dy = tgt[0] - pos[0], tgt[1] - pos[1]

        at_int = (abs(pos[0] - round(pos[0])) < EPS and
                  abs(pos[1] - round(pos[1])) < EPS)
        # 仅在交点、且两轴都需前进（有轴序选择权）时才塑形
        if self.C is not None and at_int and abs(dx) > EPS and abs(dy) > EPS:
            sx, sy = int(np.sign(dx)), int(np.sign(dy))
            ix, iy = int(round(pos[0])), int(round(pos[1]))
            cov_x = self.C[ix + sx, iy]    # 先走 x 一格到达的交点
            cov_y = self.C[ix, iy + sy]    # 先走 y 一格到达的交点
            if cov_x >= cov_y:             # 并列取 x（= 基线行为）
                return np.array([sx, 0], np.float32)
            return np.array([0, sy], np.float32)

        # 交点外（边中段，轴已锁定）或单轴剩余：照常输出对角意图，env 会沿当前边前进
        return np.sign(tgt - pos).astype(np.float32)


def make_act(mode='point', **_):
    """交付工厂：默认 mode='point'（A 段选出的最优塑形参数，但 B 段验证为负，
    结论 REJECT——若要基线行为传 mode='baseline'）。返回 act(env)。"""
    pol = PathShapePolicy(mode=mode)
    return pol.act


def _run(mode, seg):
    return evaluate(lambda: make_act(mode=mode), seg=seg)


if __name__ == '__main__':
    base = {'A': 33.99, 'B': 34.24}
    print('== 机制3 路径塑形：seg A 选参 ==')
    resA = {}
    for mode in ('baseline', 'point', 'smooth'):
        r = _run(mode, 'A')
        resA[mode] = r
        d = r['per_min'] - base['A']
        print(f"  {mode:9s} seg A: {r['per_min']:.2f} 分/分钟 "
              f"(拾取 {r['picked']}/600, Δ基线 {d:+.2f})")

    best = max(('point', 'smooth'), key=lambda m: resA[m]['per_min'])
    print(f'\n最优塑形模式（A 上）= {best}，在 seg B 验证一次:')
    rB = _run(best, 'B')
    print(f"  {best:9s} seg B: {rB['per_min']:.2f} 分/分钟 "
          f"(拾取 {rB['picked']}/293, Δ基线 {rB['per_min'] - base['B']:+.2f})")
    # 同时给出 baseline 在 B 做对照（不用于选参）
    rBb = _run('baseline', 'B')
    print(f"  baseline  seg B: {rBb['per_min']:.2f} 分/分钟 (对照)")
