# -*- coding: utf-8 -*-
"""
任务2 手工策略调参 harness（严格数据纪律版）
====================================
！！纪律：测试协议从第 894 条投放开始（1-based），因此本 harness 只暴露
前 893 条事件用于机制实验与调参：
  - seg A = 第 1..600 条   → 用来选参数
  - seg B = 第 601..893 条 → 只用来验证（选完参数后报告一次，不准回头改参）
最终对比（894 起 + 纯验证段）由主流程统一执行，实验代码不得触碰。

用法（作为库）:
    from tune_harness import seg_events, evaluate, baseline_act
    res = evaluate(make_act, seg='A')   # make_act() -> act(env)->action 的工厂
直接运行打印 baseline（现行 HeuristicPolicy）在 A/B 的成绩:
    D:/conda/envs/AI/python.exe task2/tune_harness.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bait_env import BaitGridEnv                      # noqa: E402
from heuristic_policy import HeuristicPolicy         # noqa: E402


# ---- 连续回放环境（原 pk_task2 遗留；pk_task2 已重写为纯整秒赛马场，
#      归档实验（variants/）仍依赖连续口径，故迁移至此） ----
def load_input(path=None):
    """返回事件数组 [t, x, y, v]（env 坐标 0..9）。path=None 用题目附件。"""
    from bait_env import load_events
    if path is None:
        return load_events()
    import pandas as pd
    df = pd.read_excel(path) if path.lower().endswith(('.xlsx', '.xls')) \
        else pd.read_csv(path)
    a = df.iloc[:, :4].to_numpy(float)
    a[:, 1] -= 1; a[:, 2] -= 1
    return a


class ReplayEnv(BaitGridEnv):
    """回放给定事件序列的 start_idx 条及以后，直到序列播完且场上无食饵。"""

    def __init__(self, events, start_idx=0):
        super().__init__(split='train', episode_len=1e9, synthetic=False, seed=0)
        ev = np.asarray(events, float)[start_idx:].copy()
        if len(ev):
            ev[:, 0] -= ev[0, 0] - 3.0    # 首枚移到 t=3s，给机器人起步时间
        self._replay_ev = ev
        self.start_idx = start_idx

    def _episode_events(self):
        return self._replay_ev.copy()

    def finished(self):
        return self.next_ev >= len(self.events) and not self.active

TEST_START = 893          # 0-based：events[893] 即第894条，此后为测试段，禁止使用
SEG_A_END = 600           # A = [0:600)，B = [600:893)

_EV = None


def _all_tuning_events():
    global _EV
    if _EV is None:
        _EV = load_input()[:TEST_START].copy()
    return _EV


def seg_events(seg='A'):
    ev = _all_tuning_events()
    if seg == 'A':
        return ev[:SEG_A_END]
    if seg == 'B':
        return ev[SEG_A_END:]
    if seg == 'all':
        return ev
    raise ValueError(seg)


def evaluate(make_act, seg='A', events=None):
    """make_act: 无参工厂，返回 act(env)->action。每次评测新建策略实例，防状态泄漏。
    返回 dict(score, picked, minutes, per_min)。"""
    ev = seg_events(seg) if events is None else np.asarray(events, float)
    env = ReplayEnv(ev, 0)
    env.reset()
    act = make_act()
    while not env.finished():
        env.step(act(env))
    minutes = env.t / 60.0
    return dict(score=float(env.score), picked=int(env.picked), minutes=minutes,
                per_min=float(env.score / minutes) if minutes > 0 else 0.0)


def baseline_act():
    """现行手工策略（HeuristicPolicy）的 act 工厂。"""
    pol = HeuristicPolicy()
    return pol.act


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    for seg in ('A', 'B'):
        r = evaluate(baseline_act, seg=seg)
        print(f'baseline seg {seg}: 得分 {r["score"]:.0f} | 拾取 {r["picked"]} | '
              f'{r["minutes"]:.1f} min | {r["per_min"]:.2f} 分/分钟')
