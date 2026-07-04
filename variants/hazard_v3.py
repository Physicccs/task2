# -*- coding: utf-8 -*-
"""
hazard_v3 —— v2 + 间隔风险率调制的选择性放弃（DeepSeek 建议中唯一真实新信号）
====================================================================
统一赛马接口：RobotPolicy.act(self, t, active, pos=None) → (dx, dy)

DeepSeek"统治度"框架中，唯一在既有事实体系下仍可能有增益的成分是：
间隔非指数（round(N(6.045,1.878)) clamp[0,12]）⇒ 风险率随"距上一枚
出现的时间 s"上升 ⇒ 追远处低分食饵的机会成本随 s 增大。
本策略 = heuristic_v2 全套（驻守(3,7) + 截止期排列调度 + 立即回撤），
仅在候选集过滤上加一条风险率调制的放弃规则：

    放弃食饵 b ⟺ v_b < THETA · extra(b) · P(下一枚 ≤ 往返时长 | 已等 s)

其中 extra(b) = d(pos,b) + d(b,WAIT) − d(pos,WAIT) 为绕行代价（步）。
THETA=0 时严格退化为 v2。THETA 经 tune_dominance.py 在训练种子上选定。

自检：D:/conda/envs/AI/python.exe task2/variants/hazard_v3.py --selftest
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK2 = os.path.dirname(_HERE)
for _p in (_HERE, _TASK2, os.path.join(_TASK2, '算法库')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dominance_policy import p_next_within, WAIT_POINT, LIFETIME, GRID  # noqa: E402

_V2_PATH = os.path.join(_TASK2, '算法库', 'heuristic_v2.py')
_spec = importlib.util.spec_from_file_location('hv2_base', _V2_PATH)
_hv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hv2)


class RobotPolicy(_hv2.RobotPolicy):
    """v2 + 风险率调制选择性放弃。THETA=0 ⇔ 与 v2 逐步等价。"""

    THETA = 0.5

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last_spawn = 0

    def _best_first_target(self, t, active):
        th = type(self).THETA
        if th > 0:
            s = max(0, t - self.last_spawn)
            kept = []
            for b in active:
                d_in = abs(self.pos[0] - b[1]) + abs(self.pos[1] - b[2])
                extra = (d_in + abs(b[1] - WAIT_POINT[0]) + abs(b[2] - WAIT_POINT[1])
                         - abs(self.pos[0] - WAIT_POINT[0])
                         - abs(self.pos[1] - WAIT_POINT[1]))
                trip = d_in + abs(b[1] - WAIT_POINT[0]) + abs(b[2] - WAIT_POINT[1])
                if b[3] >= th * extra * p_next_within(s, trip):
                    kept.append(b)
            active = kept
        return super()._best_first_target(t, active)

    def act(self, t, active, pos=None):
        if active:
            self.last_spawn = max(self.last_spawn, max(b[0] for b in active))
        return super().act(t, active, pos=pos)


# ---------------- 自检 ----------------
def _selftest():
    import bait_gen
    from robot_policy import load_events
    real = load_events(os.path.join('input', '202607032000建模复赛题附件.xlsx'))
    ev = bait_gen.generate(10, 99, real)

    # THETA=0 必须与 v2 逐步等价
    type_v3 = RobotPolicy
    old = type_v3.THETA
    type_v3.THETA = 0
    try:
        p3, p2 = type_v3(), _hv2.RobotPolicy()
        pos3, pos2 = (5, 5), (5, 5)
        for t in range(0, ev[-1][0] + LIFETIME + 1):
            act3 = [b for b in ev if b[0] <= t < b[0] + LIFETIME + 1 and (b[1], b[2]) != pos3]
            act2 = [b for b in ev if b[0] <= t < b[0] + LIFETIME + 1 and (b[1], b[2]) != pos2]
            s3, s2 = p3.act(t, list(act3), pos=pos3), p2.act(t, list(act2), pos=pos2)
            assert s3 == s2, f't={t}: v3(θ=0) {s3} != v2 {s2}'
            pos3 = (max(1, min(GRID, pos3[0] + s3[0])), max(1, min(GRID, pos3[1] + s3[1])))
            pos2 = (max(1, min(GRID, pos2[0] + s2[0])), max(1, min(GRID, pos2[1] + s2[1])))
    finally:
        type_v3.THETA = old

    # THETA>0 时动作仍全部合法
    pol, pos = RobotPolicy(), (5, 5)
    for t in range(0, ev[-1][0] + LIFETIME + 1):
        act = [b for b in ev if b[0] <= t < b[0] + LIFETIME + 1 and (b[1], b[2]) != pos]
        dx, dy = pol.act(t, list(act), pos=pos)
        assert (dx, dy) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        pos = (max(1, min(GRID, pos[0] + dx)), max(1, min(GRID, pos[1] + dy)))
    print(f'selftest OK：θ=0 与 v2 逐步等价 | θ={old} 动作合法')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
