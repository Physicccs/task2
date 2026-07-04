# -*- coding: utf-8 -*-
"""
dominance_policy —— "统治度势场"策略（DeepSeek 启发的善意最强实现）
====================================================================
统一赛马接口：RobotPolicy.act(self, t, active, pos=None) → (dx, dy)

来源：外部建议（DeepSeek）提出以"统治度" D(b)=Σ_{d(x,b)≤3} E[reward_x]
为势场做滚动决策，并加"当前存活食饵紧迫度"与"间隔风险率"两项修正。
本实现把该框架落到整秒仿真口径（每秒一格、位置相等即得分、第 3 秒整
踩到算得分），并将所有成分做成可调权重，避免"实现成稻草人"：

  J(b) = W_POT  · cov(b)                     覆盖率势场（修正语义版统治度）
       + W_HAZ  · Φ_DS(b, s)                 DeepSeek 原味时距加权统治度
                                             Σ_x π̂(x)·E[v]/(d+1)·P(下枚 ≤ 3−d | 已等 s)
       + W_CATCH· Σ_{可拾} v/(d(b,x)+1)      可拾食饵引力（截止期过滤）
       + W_URG  · Σ v·max(0, 龄−d)/(d+1)     DeepSeek 原味"紧迫度"项
       - W_HOME · P(下枚 ≤ 3 | 已等 s)·d(b, WAIT)   风险率调制回撤引力

每秒对 5 个候选动作（原地/四邻）取 J 最大者；平局依次按
"距最近可拾食饵更近 → 距等待点更近 → 原地"打破。

统计量来源：全部读取任务1权威基准
  output/数据分析/最终文件/task1_verification_details.json
（间隔正态 μ/σ、间隔支撑 [min,max]、E[v]、π̂=spatial.grid 计数归一化；
 等待点 = cov 的 argmax，由数据得出而非硬编码）。

权重经 tune_dominance.py 在训练种子(7000–7009)上选定后冻结为类默认值。
自检：D:/conda/envs/AI/python.exe task2/variants/dominance_policy.py --selftest
（从项目根目录运行）
"""
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK2 = os.path.dirname(_HERE)
for _p in (_TASK2, os.path.join(_TASK2, '算法库')):    # robot_policy.py 已移入算法库/
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

GRID = 10
LIFETIME = 3
START_POS = (5, 5)
VERIF_JSON = 'output/数据分析/最终文件/task1_verification_details.json'
INPUT = 'input/202607032000建模复赛题附件.xlsx'

_CELLS = [(x, y) for x in range(1, GRID + 1) for y in range(1, GRID + 1)]


def _load_verif():
    if not os.path.exists(VERIF_JSON):
        raise FileNotFoundError(
            f'需要任务1基准数据 {VERIF_JSON}（请从项目根目录运行）')
    with open(VERIF_JSON, encoding='utf-8') as f:
        return json.load(f)

_V = _load_verif()
GAP_MU = _V['gap_fits']['normal']['mu']            # 6.044971…
GAP_SIGMA = _V['gap_fits']['normal']['sigma']      # 1.878468…
GAP_MIN = int(_V['gaps']['min'])                   # 0
GAP_MAX = int(_V['gaps']['max'])                   # 12
EV_MEAN = _V['basic']['v_mean']                    # 8.740347…
_N = _V['basic']['n']                              # 1787
# spatial.grid[x-1][y-1] = 交点 (x,y) 落点计数（已用 hotspots 对齐验证）
_PI = {(x, y): _V['spatial']['grid'][x - 1][y - 1] / _N for (x, y) in _CELLS}
assert abs(sum(_PI.values()) - 1.0) < 1e-9, 'π̂ 未归一化'


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _gap_pmf():
    """间隔 ~ round(N(μ,σ)) clamp[GAP_MIN, GAP_MAX] 的 pmf。"""
    pmf = {}
    for k in range(GAP_MIN, GAP_MAX + 1):
        lo = -1e9 if k == GAP_MIN else k - 0.5
        hi = 1e9 if k == GAP_MAX else k + 0.5
        pmf[k] = _phi((hi - GAP_MU) / GAP_SIGMA) - _phi((lo - GAP_MU) / GAP_SIGMA)
    s = sum(pmf.values())
    return {k: v / s for k, v in pmf.items()}

_PMF = _gap_pmf()


def p_next_within(s, w):
    """自上一枚出现已等 s 秒且未出现，未来 w 秒内出现下一枚的概率。"""
    if w <= 0:
        return 0.0
    tail = sum(p for k, p in _PMF.items() if k >= s + 1)
    if tail <= 1e-12:
        return 1.0                       # 间隔支撑已耗尽 → 必然马上出现
    hit = sum(p for k, p in _PMF.items() if s + 1 <= k <= s + w)
    return hit / tail


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# 预计算：覆盖率势场 cov(b)、等待点（cov argmax）、Φ_DS(b, s)
_COV = {b: sum(_PI[x] for x in _CELLS if _dist(b, x) <= LIFETIME) for b in _CELLS}
WAIT_POINT = max(_COV, key=_COV.get)
_PHI_DS = {}
for _s in range(0, GAP_MAX + 1):
    _PHI_DS[_s] = {
        b: sum(_PI[x] * EV_MEAN / (_dist(b, x) + 1.0)
               * p_next_within(_s, LIFETIME - _dist(b, x))
               for x in _CELLS if _dist(b, x) <= LIFETIME)
        for b in _CELLS
    }

_MOVES = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


class RobotPolicy:
    """统治度势场策略（权重可调；默认值 = tune_dominance.py 选定的最优配置）。"""

    W_POT = 50.0      # 覆盖率势场
    W_HAZ = 0.0       # DeepSeek 原味时距加权统治度
    W_CATCH = 1.0     # 可拾食饵引力
    W_URG = 0.0       # DeepSeek 原味紧迫度项
    W_HOME = 0.0      # 风险率调制回撤引力

    def __init__(self, start=START_POS):
        self.pos = tuple(start)
        self.last_spawn = 0              # 已观测到的最近一次投放时刻

    # ---- 单点评分 ----
    def _score(self, b, t, active, s):
        cfg = type(self)
        j = 0.0
        if cfg.W_POT:
            j += cfg.W_POT * _COV[b]
        if cfg.W_HAZ:
            j += cfg.W_HAZ * _PHI_DS[min(s, GAP_MAX)][b]
        if cfg.W_CATCH or cfg.W_URG:
            for ta, bx, by, v in active:
                d = abs(b[0] - bx) + abs(b[1] - by)
                if cfg.W_CATCH and d <= ta + LIFETIME - 1 - t:   # 移动后仍赶得上判定
                    j += cfg.W_CATCH * v / (d + 1.0)
                if cfg.W_URG:
                    j += cfg.W_URG * v * max(0, (t - ta) - d) / (d + 1.0)
        if cfg.W_HOME:
            j -= cfg.W_HOME * p_next_within(s, LIFETIME) * _dist(b, WAIT_POINT)
        return j

    def act(self, t, active, pos=None):
        if pos is not None:
            self.pos = tuple(pos)
        if active:
            self.last_spawn = max(self.last_spawn, max(b[0] for b in active))
        s = max(0, t - self.last_spawn)

        catchable = [(ta, bx, by, v) for ta, bx, by, v in active
                     if abs(self.pos[0] - bx) + abs(self.pos[1] - by)
                     <= ta + LIFETIME - t]
        best, best_key = None, None
        for dx, dy in _MOVES:
            b = (max(1, min(GRID, self.pos[0] + dx)),
                 max(1, min(GRID, self.pos[1] + dy)))
            j = self._score(b, t, active, s)
            near = min((abs(b[0] - c[1]) + abs(b[1] - c[2]) for c in catchable),
                       default=0)
            key = (-j, near, _dist(b, WAIT_POINT), (dx, dy) != (0, 0))
            if best_key is None or key < best_key:
                best_key, best = key, (dx, dy)
        dx, dy = best
        self.pos = (max(1, min(GRID, self.pos[0] + dx)),
                    max(1, min(GRID, self.pos[1] + dy)))
        return dx, dy


# ---------------- 自检 ----------------
def _selftest():
    import bait_gen
    from robot_policy import load_events

    assert abs(sum(_PMF.values()) - 1.0) < 1e-9
    assert p_next_within(0, GAP_MAX) > 0.999 and p_next_within(5, 0) == 0.0
    assert 0.0 <= p_next_within(3, 3) <= 1.0
    assert WAIT_POINT == (3, 7), f'覆盖率 argmax 应为 (3,7)，得到 {WAIT_POINT}'

    # π̂ 与题目附件原始计数交叉核验（JSON spatial.grid 索引方向）
    real = load_events(INPUT)
    cnt = {}
    for _, x, y, _ in real:
        cnt[(int(x), int(y))] = cnt.get((int(x), int(y)), 0) + 1
    assert len(real) == _N, f'事件数不符 {len(real)} != {_N}'
    for c in _CELLS:
        assert abs(_PI[c] - cnt.get(c, 0) / _N) < 1e-12, f'π̂ 与附件计数不符 @ {c}'

    ev = bait_gen.generate(10, 99, real)
    pol = RobotPolicy()
    pos, seen = START_POS, 0
    for t in range(0, ev[-1][0] + LIFETIME + 1):
        act = [b for b in ev if b[0] <= t < b[0] + LIFETIME + 1
               and (b[1], b[2]) != pos]
        dx, dy = pol.act(t, act, pos=pos)
        assert (dx, dy) in _MOVES, f'非法动作 {(dx, dy)}'
        pos = (max(1, min(GRID, pos[0] + dx)), max(1, min(GRID, pos[1] + dy)))
        assert 1 <= pos[0] <= GRID and 1 <= pos[1] <= GRID
        seen += 1
    print(f'selftest OK：JSON 常数 μ={GAP_MU:.4f} σ={GAP_SIGMA:.4f} E[v]={EV_MEAN:.4f} | '
          f'π̂ 与附件逐格一致 | cov*={_COV[WAIT_POINT]:.4f} @ {WAIT_POINT} | '
          f'{seen} 步动作全部合法 | P(下枚≤3|已等5)={p_next_within(5, 3):.3f}')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
