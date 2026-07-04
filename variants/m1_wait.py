# -*- coding: utf-8 -*-
"""
M1 机制实验：等待点与回撤目标优化（静态全网格搜索 / 条件回撤 / 半格点证伪）
================================================================================
【最优参数与成绩】
  机制: 静态等待点（条件回撤与半格点均被证伪，见下）
  参数: wait_point=env(2,6)（表格(3,7)）, lam=None, cands=None —— 与基线相同点
  seg A: 33.99 分/分钟   seg B: 34.24 分/分钟（基线 A 33.99 / B 34.24 → 无增益）

【实验结论（seg A）】
  1) 静态全网格 100 交点快筛（整数秒模拟器）前3: (2,6) 36.62 > (2,7) 36.22 > (3,7) 34.72；
     连续环境确认: (2,6) 33.99 > (2,7) 32.48 > (3,7) 31.74 → 基线等待点已是全局最优。
     注意 (2,6) 与 (2,7) 在 seg A 覆盖率精确并列（0.4283），但实测差 1.5 分/分钟：
     覆盖率并列时靠南的点期望距离更小、且更贴近场地中部（回撤/追击顺带覆盖更好）。
  2) 条件回撤 argmax[cov(w)−λ·d(pos,w)]：快筛 λ∈{0.005..0.08}×cands∈{5..40} 最好 36.60，
     不超静态 36.62；连续环境确认 (λ=0.005,c=5) 32.80、(λ=0.02,c=10) 32.97，均劣于 33.99
     → 拒绝。机制层原因：覆盖率地形是单峰（唯一主热点），离开峰顶的"就近"候选点
     覆盖损失 > 路程节省；且 λ 使机器人在覆盖并列/接近的点之间漂移，丢掉唯一最优点的优势。
  3) 半格边中点: (2,6.5) 33.38、(2.5,6) 23.08、(2,5.5) 27.60，全部劣于 33.99 → 证伪。
     曼哈顿可达球从边中点出发对整点格半径实际亏 0.5m，覆盖必降。

机制说明
--------
追击逻辑与基线 HeuristicPolicy 逐行相同（带截止期的价值前5全排列调度，直接委托
HeuristicPolicy._best_first_target）；只改"无可达食饵时去哪儿等"：
  1) 静态等待点 wait_point（env 坐标，全网格 100 交点扫描选出）；
  2) 条件回撤（lam 非 None 时启用）：回撤目标 = argmax_w [cov(w) − lam·d(pos,w)]，
     w 取 seg A 经验落点分布下曼哈顿 3m 覆盖率前 cands 名的交点；lam→0 退化为全局最优点；
  3) 半格等待点（x 或 y 为 .5 的边中点）作为对照实验（理论上曼哈顿覆盖亏半径，预期更差）。

数据纪律：只用 tune_harness 暴露的前 893 条（seg A=前600 选参，seg B=601..893 验证一次）。
覆盖率/经验分布一律只从 seg_events('A') 估计。

用法（从项目根目录 E:\\working_flow\\program 运行）：
  D:/conda/envs/AI/python.exe task2/variants/m1_wait.py scan_static   # 快筛100交点
  D:/conda/envs/AI/python.exe task2/variants/m1_wait.py confirm_static
  D:/conda/envs/AI/python.exe task2/variants/m1_wait.py scan_cond     # λ×cands 快筛
  D:/conda/envs/AI/python.exe task2/variants/m1_wait.py confirm_cond
  D:/conda/envs/AI/python.exe task2/variants/m1_wait.py half          # 半格点证伪
  D:/conda/envs/AI/python.exe task2/variants/m1_wait.py final_B       # 最优参数 seg B 一次
交付接口： make_act(wait_point=..., lam=..., cands=...) → act(env)
"""
import sys
import os
from itertools import permutations

sys.path.insert(0, 'task2')
sys.path.insert(0, os.path.join('task2', 'variants'))

import numpy as np                                    # noqa: E402
from heuristic_policy import HeuristicPolicy          # noqa: E402
from tune_harness import seg_events, evaluate         # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

LIFE = 3          # 存活秒数（整数秒模拟器用）

# ---- 最优参数（实验后回填；make_act 默认值即最终交付配置） ----
BEST_WAIT = (2.0, 6.0)     # env 坐标
BEST_LAM = None            # None = 静态等待点机制
BEST_CANDS = None


# ================= 覆盖率表（只用 seg A） =================
def coverage_table():
    """cov[wx,wy] = seg A 经验落点分布下、曼哈顿 3m 可达覆盖率（env 坐标交点）。"""
    ev = seg_events('A')
    cells = (ev[:, 1] * 10 + ev[:, 2]).astype(int)
    p = np.bincount(cells, minlength=100) / len(cells)
    cx, cy = np.arange(100) // 10, np.arange(100) % 10
    cov = np.zeros((10, 10))
    for wx in range(10):
        for wy in range(10):
            cov[wx, wy] = p[np.abs(cx - wx) + np.abs(cy - wy) <= LIFE].sum()
    return cov


def top_candidates(cov, n):
    """覆盖率前 n 名交点（降序；并列按坐标序，保证确定性）。"""
    flat = sorted(((-cov[x, y], x, y) for x in range(10) for y in range(10)))
    return [(x, y) for _, x, y in flat[:n]]


# ================= 整数秒快速模拟器（复制 robot_policy.simulate 思路，env 坐标 0..9） =================
def _best_first_int(pos, t, active):
    """带截止期排列调度（整数版，与 HeuristicPolicy 同口径），返回首目标或 None。"""
    cand = [b for b in active
            if abs(pos[0] - b[1]) + abs(pos[1] - b[2]) <= b[0] + LIFE - t]
    if not cand:
        return None
    cand = sorted(cand, key=lambda b: -b[3])[:5]
    best_v, best_T, best_first = -1.0, 1e18, None
    for perm in permutations(range(len(cand))):
        px, py, tt, val, first = pos[0], pos[1], t, 0.0, None
        for i in perm:
            ta, bx, by, v = cand[i]
            eta = tt + abs(px - bx) + abs(py - by)
            if eta <= ta + LIFE:
                val += v
                tt, px, py = eta, bx, by
                if first is None:
                    first = (bx, by)
        if first and (val > best_v or (val == best_v and tt < best_T)):
            best_v, best_T, best_first = val, tt, first
    return best_first


def fast_sim(events, wait_fn):
    """整数秒回放（env 坐标）。wait_fn(pos)->空闲时回撤目标交点。
    口径与 robot_policy.simulate 一致：首枚移到 t=3；拾取判定先于过期。"""
    ev = sorted(map(tuple, np.asarray(events, float)), key=lambda e: e[0])
    shift = ev[0][0] - 3
    ev = [(e[0] - shift, int(e[1]), int(e[2]), e[3]) for e in ev]
    pos, active, nxt = (4, 4), [], 0
    t, score, picked = 0, 0.0, 0
    while nxt < len(ev) or active:
        while nxt < len(ev) and ev[nxt][0] <= t:
            active.append(ev[nxt]); nxt += 1
        for b in [b for b in active if (b[1], b[2]) == pos]:
            score += b[3]; picked += 1
            active.remove(b)
        active = [b for b in active if t - b[0] < LIFE]
        tgt = _best_first_int(pos, t, active) or wait_fn(pos)
        ddx, ddy = tgt[0] - pos[0], tgt[1] - pos[1]
        if ddx == 0 and ddy == 0:
            step = (0, 0)
        elif abs(ddx) >= abs(ddy):
            step = (1 if ddx > 0 else -1, 0)
        else:
            step = (0, 1 if ddy > 0 else -1)
        pos = (pos[0] + step[0], pos[1] + step[1])
        t += 1
    minutes = t / 60
    return dict(score=score, picked=picked,
                per_min=score / minutes if minutes else 0.0)


# ================= 连续环境策略（交付物） =================
class M1Policy:
    """追击=基线排列调度；空闲回撤目标可配置（静态点 / 条件回撤 argmax[cov−λd]）。"""

    def __init__(self, wait_point=BEST_WAIT, lam=None, cands=None):
        # 复用基线的排列调度；train_ev 传 seg A 保证只用许可数据（等待点随后覆盖）
        self.h = HeuristicPolicy(train_ev=seg_events('A'))
        self.wait_point = np.array(wait_point, float)
        self.lam = lam
        if lam is not None:
            cov = coverage_table()
            pts = top_candidates(cov, cands or 10)
            self.c_pts = np.array(pts, float)                       # 覆盖率降序
            self.c_cov = np.array([cov[x, y] for x, y in pts])

    def _retreat_target(self, pos):
        if self.lam is None:
            return self.wait_point
        d = np.abs(self.c_pts[:, 0] - pos[0]) + np.abs(self.c_pts[:, 1] - pos[1])
        return self.c_pts[int(np.argmax(self.c_cov - self.lam * d))]

    def act(self, env):
        tgt = self.h._best_first_target(env)
        if tgt is None:
            tgt = self._retreat_target(env.pos)
        return np.sign(tgt - env.pos).astype(np.float32)


def make_act(wait_point=BEST_WAIT, lam=BEST_LAM, cands=BEST_CANDS):
    """无参调用即最终交付配置（供 tune_harness.evaluate 直接使用）。"""
    pol = M1Policy(wait_point=wait_point, lam=lam, cands=cands)
    return pol.act


# ================= 实验驱动 =================
def scan_static():
    """快筛：100 个交点全扫（seg A，整数秒模拟器），列前 8 名。"""
    ev = seg_events('A')
    cov = coverage_table()
    res = []
    for wx in range(10):
        for wy in range(10):
            r = fast_sim(ev, lambda pos, w=(wx, wy): w)
            res.append((r['per_min'], wx, wy, r['picked']))
    res.sort(reverse=True)
    print('== 静态等待点快筛（整数秒模拟器, seg A）前 8 名 ==')
    print(f'{"env点":>8} {"表格点":>8} {"分/分钟":>9} {"拾取":>5} {"cov3m":>7}')
    for pm, wx, wy, pk in res[:8]:
        print(f'  ({wx},{wy})   ({wx+1},{wy+1})  {pm:9.2f} {pk:5d} {cov[wx, wy]:7.4f}')
    print(f'（对照）基线点 env(2,6): '
          f'{[r for r in res if (r[1], r[2]) == (2, 6)][0][0]:.2f} 分/分钟')
    return res


def confirm_static(points):
    """把候选点放进连续环境 evaluate(seg='A') 确认。"""
    print('== 静态等待点连续环境确认 (seg A) ==')
    for (wx, wy) in points:
        r = evaluate(lambda w=(float(wx), float(wy)): make_act(wait_point=w, lam=None),
                     seg='A')
        print(f'  env({wx},{wy}) 表格({wx+1},{wy+1}): {r["per_min"]:.2f} 分/分钟 '
              f'(得分 {r["score"]:.0f}, 拾取 {r["picked"]})')


def scan_cond():
    """条件回撤快筛：λ×cands 网格（整数秒模拟器, seg A）。"""
    ev = seg_events('A')
    cov = coverage_table()
    print('== 条件回撤快筛（整数秒模拟器, seg A）分/分钟 ==')
    lams = [0.005, 0.01, 0.02, 0.04, 0.08]
    ns = [5, 10, 20, 40]
    print(f'{"lam\\cands":>10} ' + ' '.join(f'{n:>7}' for n in ns))
    best = (-1, None, None)
    for lam in lams:
        row = []
        for n in ns:
            pts = top_candidates(cov, n)
            cv = np.array([cov[x, y] for x, y in pts])
            arr = np.array(pts, float)

            def wait_fn(pos, arr=arr, cv=cv, lam=lam):
                d = np.abs(arr[:, 0] - pos[0]) + np.abs(arr[:, 1] - pos[1])
                w = arr[int(np.argmax(cv - lam * d))]
                return (int(w[0]), int(w[1]))

            r = fast_sim(ev, wait_fn)
            row.append(r['per_min'])
            if r['per_min'] > best[0]:
                best = (r['per_min'], lam, n)
        print(f'{lam:>10.3f} ' + ' '.join(f'{v:7.2f}' for v in row))
    print(f'最优: lam={best[1]}, cands={best[2]}, {best[0]:.2f} 分/分钟')
    return best


def confirm_cond(combos, ref_wait):
    """条件回撤连续环境确认，并与静态最优点对照。"""
    print('== 条件回撤连续环境确认 (seg A) ==')
    for lam, n in combos:
        r = evaluate(lambda l=lam, c=n: make_act(wait_point=ref_wait, lam=l, cands=c),
                     seg='A')
        print(f'  lam={lam}, cands={n}: {r["per_min"]:.2f} 分/分钟 '
              f'(得分 {r["score"]:.0f}, 拾取 {r["picked"]})')


def half_points(points):
    """半格等待点（边中点）连续环境测试。"""
    print('== 半格等待点连续环境测试 (seg A) ==')
    for (wx, wy) in points:
        r = evaluate(lambda w=(float(wx), float(wy)): make_act(wait_point=w, lam=None),
                     seg='A')
        print(f'  env({wx},{wy}): {r["per_min"]:.2f} 分/分钟 '
              f'(得分 {r["score"]:.0f}, 拾取 {r["picked"]})')


def final_B():
    """最优机制+参数在 seg B 验证一次（不准回头改参）。"""
    r = evaluate(make_act, seg='B')
    print(f'== 最终 seg B 验证（wait_point={BEST_WAIT}, lam={BEST_LAM}, '
          f'cands={BEST_CANDS}）==')
    print(f'  seg B: {r["per_min"]:.2f} 分/分钟 (得分 {r["score"]:.0f}, '
          f'拾取 {r["picked"]}, {r["minutes"]:.1f} min)')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'scan_static'
    if stage == 'scan_static':
        scan_static()
    elif stage == 'confirm_static':
        pts = [tuple(int(v) for v in a.split(',')) for a in sys.argv[2:]] \
            or [(2, 6), (2, 7), (3, 7)]
        confirm_static(pts)
    elif stage == 'scan_cond':
        scan_cond()
    elif stage == 'confirm_cond':
        combos = [(float(a.split(',')[0]), int(a.split(',')[1])) for a in sys.argv[2:]]
        confirm_cond(combos, BEST_WAIT)
    elif stage == 'half':
        pts = [tuple(float(v) for v in a.split(',')) for a in sys.argv[2:]] \
            or [(2.0, 6.5), (2.5, 7.0), (2.0, 5.5)]
        half_points(pts)
    elif stage == 'final_B':
        final_B()
    else:
        raise SystemExit(f'未知 stage: {stage}')
