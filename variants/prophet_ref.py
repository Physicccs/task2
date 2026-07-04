# -*- coding: utf-8 -*-
"""
1步先知参照基准 ProphetPolicy —— 衡量"完美预判站位"的价值上限（仅作标尺，非交付物）
====================================================================================
【成绩】无可调参数（make_act() 即最优）：
  seg A: 65.27 分/分钟（拾取 436/600，基线 33.99，预判增益 +31.28）
  seg B: 65.88 分/分钟（拾取 210/293，基线 34.24，预判增益 +31.64）
  参考：A/B 全部食饵总价值 ≈ 86.4 / 90.5 分/分钟 → 先知吃到约 75%，基线约 39%。

与 HeuristicPolicy 完全相同（追击用同一套带截止期的排列调度 _best_first_target），
唯一区别在"空闲（无可达在场食饵）"时：
  - 基线：回固定"覆盖率最优等待点" env(2,6)。
  - 先知：偷看未来一条事件（env.events[env.next_ev]，ReplayEnv.events 即全序列），
          提前沿网格线走向"下一枚将出现的食饵"的落点并等在那里（遵守 1m/s、不能瞬移）。

【拾取机制的坑（务必理解）】bait_env 的 _pickups 只在机器人"扫过的线段"经过食饵点时得分；
零动作不产生线段 → 停在落点正上方的机器人永远拾取不到刚生成的食饵。
故先知不能停在落点 L 正上方，而是预置在 L 的 1m 邻域（dist≤1）等待；食饵一生成即被
_best_first_target 判为可达，机器人沿单轴扫过 L 完成拾取（≤1s，远在 3s 存活期内）。

解读：该分数 ≈ "把等待站位站到完美位置"的在线不可达上界。
它与基线（A 33.99 / B 34.24）之差 = 位置预判信息的全部价值；真实在线策略只能通过更优的
静态/条件站位吃掉其中一小部分。

数据纪律：只用 tune_harness 暴露的前 893 条（seg A=前600，seg B=601..893）。
运行： D:/conda/envs/AI/python.exe task2/variants/prophet_ref.py
"""
import sys
import os

sys.path.insert(0, 'task2')
sys.path.insert(0, os.path.join('task2', 'variants'))

import numpy as np  # noqa: E402
from bait_env import GRID, EPS  # noqa: E402
from heuristic_policy import HeuristicPolicy  # noqa: E402
from tune_harness import ReplayEnv  # noqa: E402
from tune_harness import seg_events  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class ProphetPolicy:
    """作弊参照：追击逻辑与 HeuristicPolicy 完全一致，空闲时预置到下一枚食饵落点的 1m 邻域。"""

    def __init__(self, train_ev=None):
        self.h = HeuristicPolicy(train_ev)     # 复用等待点与排列调度，保证追击行为一致
        self.wait_point = self.h.wait_point

    @staticmethod
    def _next_landing(env):
        """偷看未来一条事件：ReplayEnv.events 是全序列，next_ev 指向下一枚尚未投放的食饵。"""
        if env.next_ev < len(env.events):
            e = env.events[env.next_ev]
            return np.array([e[1], e[2]], float)
        return None

    def act(self, env):
        # 1) 有可达在场食饵 → 与基线完全相同的带截止期排列调度，奔向首目标
        tgt = self.h._best_first_target(env)
        if tgt is not None:
            return np.sign(tgt - env.pos).astype(np.float32)

        # 2) 空闲：偷看下一枚将出现食饵的落点 L，预置到其 1m 邻域等待
        L = self._next_landing(env)
        if L is None:                                   # 没有未来信息 → 退回覆盖率最优等待点
            return np.sign(self.wait_point - env.pos).astype(np.float32)

        pos = env.pos
        d = abs(pos[0] - L[0]) + abs(pos[1] - L[1])
        if d < EPS:
            # 恰好站在落点正上方（拾取不到）→ 挪到一个合法邻点，以便食饵生成后能扫回 L
            for dv in ([1, 0], [-1, 0], [0, 1], [0, -1]):
                q = L + np.array(dv, float)
                if -EPS <= q[0] <= GRID + EPS and -EPS <= q[1] <= GRID + EPS:
                    return np.sign(q - pos).astype(np.float32)
            return np.zeros(2, np.float32)
        if d <= 1.0 + EPS:
            return np.zeros(2, np.float32)               # 已在 1m 邻域 → 原地等待，生成即扫入
        return np.sign(L - pos).astype(np.float32)       # 尚远 → 沿网格线逼近落点（物理限速）


def make_act(**_params):
    """交付约定：无参工厂 → act(env)。先知无可调参数（作弊标尺）。"""
    pol = ProphetPolicy()
    return pol.act


def run_seg(seg):
    """自写回放循环（复制 tune_harness.evaluate 的循环），策略内部读 env.events/next_ev 偷看未来。"""
    ev = seg_events(seg)
    env = ReplayEnv(ev, 0)
    env.reset()
    act = make_act()
    while not env.finished():
        env.step(act(env))
    minutes = env.t / 60.0
    return dict(score=float(env.score), picked=int(env.picked), minutes=minutes,
                per_min=float(env.score / minutes) if minutes > 0 else 0.0)


if __name__ == '__main__':
    base = {'A': 33.99, 'B': 34.24}
    print('== 1步先知参照基准（作弊标尺）==')
    for seg in ('A', 'B'):
        r = run_seg(seg)
        gain = r['per_min'] - base[seg]
        print(f'先知 seg {seg}: 得分 {r["score"]:.0f} | 拾取 {r["picked"]} | '
              f'{r["minutes"]:.1f} min | {r["per_min"]:.2f} 分/分钟'
              f'  (基线 {base[seg]:.2f}, 预判增益 +{gain:.2f})')
