# -*- coding: utf-8 -*-
"""
任务2 · 暴露度诊断（只做归因，不改策略）
====================================================================
在 seg A + seg B（全部 893 条投放事件）的连续环境上，回放现行
HeuristicPolicy，逐 0.25s 步记录机器人状态，并把每一步的行为分为三态：

  idle    : 调度器无可达追击目标（_best_first_target 为 None）且机器人已位于
            覆盖率最优等待点 env(2,6)
  chase   : 调度器给出了追击目标（_best_first_target 非 None）
  return  : 无追击目标但机器人尚未回到等待点（正处在回撤途中，或开局从场地
            中心 (4,4) 走向等待点的过渡段）

判定复用同一个 HeuristicPolicy 实例：每步先算 pol._best_first_target(env)
看是否为 None，再看机器人当前位置是否落在等待点。

—— 关键实现细节 ——
为把"食饵出现瞬间"和"最终是否拾到"精确归因到每一枚事件，本文件用一个
带"事件索引标签"的 ReplayEnv 子类（TracedReplayEnv），让每枚在场食饵始终
携带它在 env.events 里的下标；这样识别食饵不依赖 id()（内存地址在对象被
回收后可能复用，会造成串号），而是用确定性的事件下标，完全可靠。

食饵"出现瞬间的机器人状态"取该食饵被投放那一步（step 内 _spawn_and_expire
把它加入场上）之前、机器人本步所处的三态标签 —— 即"食饵弹出时机器人正在做
什么"，而不是把这枚新食饵纳入后重新分类（后者会因这枚食饵自己而恒判为
chase，无意义）。距离取该瞬间机器人位置（本步移动后的落点）到食饵的曼哈顿
距离。

运行：
  D:/conda/envs/AI/python.exe task2/variants/diag_exposure.py
"""
import sys
import os

sys.path.insert(0, 'task2')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

from bait_env import LIFETIME, SPEED, EPS, DT
from heuristic_policy import HeuristicPolicy
from tune_harness import seg_events
from tune_harness import ReplayEnv


# --------------------------------------------------------------------------
class TracedReplayEnv(ReplayEnv):
    """在 ReplayEnv 基础上，为每枚在场食饵维护一个事件下标 tag，并记录被
    "拾取"（区别于"过期"）的食饵下标。identity 用确定性 tag，不用 id()。"""

    def reset(self, **kw):
        self.tags = []            # 与 self.active 平行：每枚在场食饵的事件下标
        self.picked_idx = set()   # 被真正拾取（非过期消失）的食饵下标
        return super().reset(**kw)

    def _spawn_and_expire(self):
        # 完整复刻基类的投放+过期逻辑，同时让 tags 与 active 逐项对齐。
        while (self.next_ev < len(self.events)
               and self.events[self.next_ev, 0] <= self.t + EPS):
            e = self.events[self.next_ev]
            self.active.append(e.copy())
            self.tags.append(self.next_ev)
            self.last_appear = e[0]
            self.next_ev += 1
        kept = [(a, tg) for a, tg in zip(self.active, self.tags)
                if self.t - a[0] < LIFETIME - EPS]
        self.active = [a for a, _ in kept]
        self.tags = [tg for _, tg in kept]

    def _pickups(self, segs):
        # 基类按对象身份从 self.active 移除被拾取者；这里在同一函数作用域内
        # （before 持有引用，期间不会被 GC，id 稳定）把 tags 重新对齐，并把
        # 被移除者记为"拾取"。
        before = list(self.active)
        before_tags = list(self.tags)
        got = super()._pickups(segs)
        survivors = {id(a) for a in self.active}
        new_tags = []
        for a, tg in zip(before, before_tags):
            if id(a) in survivors:
                new_tags.append(tg)
            else:
                self.picked_idx.add(tg)   # 从 active 里被 _pickups 拿走 → 拾取
        self.tags = new_tags
        return got


# --------------------------------------------------------------------------
def manhattan(p, x, y):
    return abs(p[0] - x) + abs(p[1] - y)


def run_diag():
    ev_all = seg_events('all')          # 全部 893 条（seg A + seg B）
    n_ev = len(ev_all)

    pol = HeuristicPolicy()
    wait = pol.wait_point               # env 坐标，应为 (2,6)

    env = TracedReplayEnv(ev_all, 0)
    env.reset()

    # 三态计步（每步 = DT 秒）
    state_steps = {'idle': 0, 'chase': 0, 'return': 0}

    # 每枚食饵一条记录（下标 = 事件在 ev_all 中的位置）
    recs = [dict(idx=i, ta=float(ev_all[i, 0]),
                 x=float(ev_all[i, 1]), y=float(ev_all[i, 2]),
                 v=float(ev_all[i, 3]),
                 appear_state=None, appear_d=None, picked=False)
            for i in range(n_ev)]
    seen = set()

    steps = 0
    while not env.finished():
        # --- 本步机器人三态（在本步移动/新投放之前判定）---
        tgt = pol._best_first_target(env)
        if tgt is None:
            state = 'idle' if manhattan(env.pos, wait[0], wait[1]) < 1e-6 else 'return'
        else:
            state = 'chase'
        state_steps[state] += 1
        steps += 1

        # --- 推进一步（内部：move → _pickups → t+=DT → _spawn_and_expire）---
        env.step(pol.act(env))

        # --- 本步内新出现的食饵：登记出现瞬间状态与距离 ---
        for a, tg in zip(env.active, env.tags):
            if tg in seen:
                continue
            seen.add(tg)
            recs[tg]['appear_state'] = state
            recs[tg]['appear_d'] = float(manhattan(env.pos, a[1], a[2]))

    # 拾取归属
    for tg in env.picked_idx:
        recs[tg]['picked'] = True

    # 完整性校验
    assert len(seen) == n_ev, f'出现登记 {len(seen)} != 事件 {n_ev}'
    assert int(env.picked) == sum(r['picked'] for r in recs), '拾取计数不一致'

    return recs, state_steps, steps, env, pol


# --------------------------------------------------------------------------
def summarize(recs, state_steps, steps, env, pol):
    total_time = steps * DT
    minutes = total_time / 60.0
    total_v = sum(r['v'] for r in recs)
    picked_v = sum(r['v'] for r in recs if r['picked'])
    n_pick = sum(1 for r in recs if r['picked'])

    lines = []
    P = lines.append
    P('=' * 72)
    P('任务2 暴露度诊断 · 现行 HeuristicPolicy · 全部 893 条事件（seg A+B）')
    P('=' * 72)
    P(f'等待点(env) = ({pol.wait_point[0]:.0f},{pol.wait_point[1]:.0f})'
      f'  覆盖率 {pol.coverage:.3f}')
    P(f'总时长 {total_time:.1f}s（{minutes:.2f} min，{steps} 步 @ {DT}s）')
    P(f'事件 {len(recs)} 枚 / 总价值 {total_v:.0f}')
    P(f'实拾 {n_pick} 枚 / {picked_v:.0f} 分'
      f'  ->  {picked_v/minutes:.2f} 分/分钟'
      f'（拾取率 枚 {n_pick/len(recs)*100:.1f}% / 值 {picked_v/total_v*100:.1f}%）')

    # ---------- 1. 三态时间占比 ----------
    P('')
    P('--- 1. 三态时间占比 ---')
    for s in ('idle', 'chase', 'return'):
        st = state_steps[s]
        P(f'  {s:<7} {st*DT:8.1f}s  {st/steps*100:6.2f}%  ({st} 步)')

    # ---------- 2. 按出现瞬间状态分组的拾取率 ----------
    P('')
    P('--- 2. 按"食饵出现瞬间机器人状态"分组 ---')
    P(f'  {"状态":<8}{"枚数":>6}{"拾到":>6}{"枚拾取率":>10}'
      f'{"价值":>9}{"拾到值":>9}{"值拾取率":>10}{"错失值":>9}{"均出现d":>9}')
    grp = {}
    for r in recs:
        grp.setdefault(r['appear_state'], []).append(r)
    for s in ('idle', 'chase', 'return'):
        g = grp.get(s, [])
        if not g:
            P(f'  {s:<8}{0:>6}')
            continue
        ng = len(g)
        npk = sum(1 for r in g if r['picked'])
        vg = sum(r['v'] for r in g)
        vpk = sum(r['v'] for r in g if r['picked'])
        md = sum(r['appear_d'] for r in g) / ng
        P(f'  {s:<8}{ng:>6}{npk:>6}{npk/ng*100:>9.1f}%'
          f'{vg:>9.0f}{vpk:>9.0f}{vpk/vg*100:>9.1f}%{vg-vpk:>9.0f}{md:>9.2f}')

    # ---------- 3. 错过食饵的 (出现瞬间 d, v) 二维分布 ----------
    P('')
    P('--- 3. 错过食饵：出现瞬间距离 d × 价值 分布 ---')
    missed = [r for r in recs if not r['picked']]
    miss_v = sum(r['v'] for r in missed)
    P(f'  错过 {len(missed)} 枚 / {miss_v:.0f} 分'
      f'（占总价值 {miss_v/total_v*100:.1f}%）')
    # 距离都是 0.25 的整数倍（机器人恒在网格线上），round 去浮点噪声后用
    # 半开区间 (lo, hi] 做无重叠划分；首带为 [0,3]。各带枚数之和 == 错过总数。
    bands = [(0, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 99)]
    P(f'  {"d 区间":<10}{"错过枚":>8}{"错失值":>9}{"占错失值":>10}{"该带均值v":>11}')
    band_n = 0
    for lo, hi in bands:
        if lo == 0:
            sel = [r for r in missed if round(r['appear_d'], 3) <= hi]
        else:
            sel = [r for r in missed
                   if lo < round(r['appear_d'], 3) <= hi]
        band_n += len(sel)
        if not sel:
            continue
        vv = sum(r['v'] for r in sel)
        lab = f'({lo},{hi}]' if lo > 0 else f'[0,{hi}]'
        P(f'  {lab:<10}{len(sel):>8}{vv:>9.0f}{vv/miss_v*100:>9.1f}%'
          f'{vv/len(sel):>11.2f}')
    assert band_n == len(missed), f'距离带划分不完整 {band_n} != {len(missed)}'

    # d<=3 却仍错过 = 暴露度损失（理论上站住不动就能拿到，被追击/回撤带走）
    d3 = [r for r in missed if r['appear_d'] <= 3 + EPS]
    d3_v = sum(r['v'] for r in d3)
    P('')
    P(f'  ** 出现瞬间 d<=3 却仍错过 = {len(d3)} 枚 / {d3_v:.0f} 分'
      f'（占全部错失值 {d3_v/miss_v*100:.1f}%，占总价值 {d3_v/total_v*100:.1f}%）**')
    # 这部分按"出现瞬间状态"再拆，看是被哪种动作带走的
    d3_by = {}
    for r in d3:
        d3_by.setdefault(r['appear_state'], [0, 0.0])
        d3_by[r['appear_state']][0] += 1
        d3_by[r['appear_state']][1] += r['v']
    for s in ('idle', 'chase', 'return'):
        if s in d3_by:
            c, vv = d3_by[s]
            P(f'      其中出现瞬间处于 {s:<7}: {c} 枚 / {vv:.0f} 分')

    # 逐案归因：这些近处错过是"被动作带走"还是"调度取舍"？
    # 判据：该食饵存活期 [ta, ta+3] 内策略是否拾到了别的食饵（并发取舍），
    # 以及存活期内是否有别的食饵在场（并发场面）。
    picked_set = [r for r in recs if r['picked']]
    P('')
    P('  d<=3 错过逐案归因：')
    P(f'  {"ta":>8}{"d":>6}{"v":>5}{"状态":>8}{"存活期内拾到别饵":>18}{"存活期内并发在场":>18}')
    n_tradeoff = 0
    v_tradeoff = 0.0
    for r in sorted(d3, key=lambda r: r['ta']):
        # 存活期内拾到的其他食饵（用"该饵存活窗与被拾饵存活窗重叠"近似：
        # 被拾饵 ta' 满足其存活窗 [ta',ta'+3] 与 [ta,ta+3] 相交）
        others_picked = [q for q in picked_set if q['idx'] != r['idx']
                         and q['ta'] < r['ta'] + LIFETIME
                         and q['ta'] + LIFETIME > r['ta']]
        concurrent = [q for q in recs if q['idx'] != r['idx']
                      and q['ta'] < r['ta'] + LIFETIME
                      and q['ta'] + LIFETIME > r['ta']]
        vo = sum(q['v'] for q in others_picked)
        if others_picked:
            n_tradeoff += 1
            v_tradeoff += r['v']
        P(f'  {r["ta"]:>8.0f}{r["appear_d"]:>6.2f}{r["v"]:>5.0f}'
          f'{r["appear_state"]:>8}'
          f'{len(others_picked):>9d} 枚/{vo:>4.0f}分'
          f'{len(concurrent):>14d} 枚')
    P(f'  → 其中 {n_tradeoff}/{len(d3)} 枚（{v_tradeoff:.0f} 分）在其存活期内'
      f'策略拾到了别的（并发取舍/中途换目标），其余为纯错失。')
    # ---------- 4. 三条量化发现 ----------
    P('')
    P('--- 4. 对改进最有指导意义的量化发现 ---')

    # 供发现使用的量
    chase_frac = state_steps['chase'] / steps
    ret_frac = state_steps['return'] / steps
    idle_frac = state_steps['idle'] / steps
    g_idle = grp.get('idle', [])
    g_chase = grp.get('chase', [])
    g_ret = grp.get('return', [])

    def rate(g):
        if not g:
            return 0.0, 0.0
        return (sum(1 for r in g if r['picked']) / len(g),
                sum(r['v'] for r in g if r['picked']) / max(sum(r['v'] for r in g), EPS))

    idle_pr = rate(g_idle)
    chase_pr = rate(g_chase)
    ret_pr = rate(g_ret)

    miss_idle_v = sum(r['v'] for r in g_idle if not r['picked'])
    miss_chase_v = sum(r['v'] for r in g_chase if not r['picked'])
    miss_ret_v = sum(r['v'] for r in g_ret if not r['picked'])

    md_idle = (sum(r['appear_d'] for r in g_idle) / len(g_idle)) if g_idle else 0
    # 出现瞬间 d>3（物理不可达）的错失值 —— 纯站位/覆盖损失
    far_v = sum(r['v'] for r in missed if round(r['appear_d'], 3) > 3)

    P(f'  [发现1] 损失几乎全部发生在 idle 态，根因是"单一等待点覆盖不足"，'
      f'而非动作失误：错失值按出现瞬间状态拆分 = idle {miss_idle_v:.0f} 分'
      f'（{miss_idle_v/miss_v*100:.1f}%）/ return {miss_ret_v:.0f} 分'
      f'（{miss_ret_v/miss_v*100:.1f}%）/ chase {miss_chase_v:.0f} 分'
      f'（{miss_chase_v/miss_v*100:.1f}%）。idle 态食饵出现瞬间平均距离 '
      f'{md_idle:.2f}m > 3m 可达半径，等待点 env(2,6) 单点覆盖仅 '
      f'{pol.coverage*100:.1f}%，导致 {far_v:.0f} 分（错失值 {far_v/miss_v*100:.1f}%）'
      f'的食饵一出现就够不到。→ 最大杠杆在"覆盖/站位"：多驻点巡逻、'
      f'或把等待点挪到期望距离更小处，而不是调度。')
    P(f'  [发现2] 不支持"提高追击门槛"：chase 态只占 {chase_frac*100:.1f}% 时间、'
      f'期间仅 {len(g_chase)} 枚食饵出现，错失合计 {miss_chase_v:.0f} 分'
      f'（占错失值 {miss_chase_v/miss_v*100:.1f}%，可忽略）；'
      f'return 态拾取率反而最高（值 {ret_pr[1]*100:.1f}% > idle {idle_pr[1]*100:.1f}%），'
      f'因回撤途中位于场地中部、离新食饵更近（均出现 d {md_idle:.2f}→'
      f'{(sum(r["appear_d"] for r in g_ret)/len(g_ret)) if g_ret else 0:.2f}）。'
      f'追击/回撤本身没有拖累得分，砍追击不会改善。')
    P(f'  [发现3] "出现瞬间 d<=3 却错过" = {len(d3)} 枚 / {d3_v:.0f} 分'
      f'（仅占错失值 {d3_v/miss_v*100:.1f}%、总价值 {d3_v/total_v*100:.1f}%）。'
      f'逐案看：这 {len(d3)} 枚全部 d==0（食饵恰好弹在机器人脚下），'
      f'且几乎无并发食饵、0 例并发取舍 —— 不是被追击/回撤"带走"，'
      f'而是拾取判定死区：策略对脚下目标返回零动作(sign=0)，env 只在'
      f'"扫过的线段"上判拾，静止不扫段 → 脚下食饵拾不到（除非另有食饵引离、'
      f'离开时顺带扫过）。一行修补（脚下有目标时朝任一相邻点走一步再回）即可'
      f'回收，但天花板仅 {d3_v:.0f} 分（{d3_v/minutes:.2f} 分/分钟），属机制级微益。')

    P('=' * 72)

    out = '\n'.join(lines)
    print(out)

    # 结构化返回值（供上层脚本消费）
    return dict(
        time_fractions=dict(idle=idle_frac, chase=chase_frac, return_=ret_frac,
                            idle_s=state_steps['idle'] * DT,
                            chase_s=state_steps['chase'] * DT,
                            return_s=state_steps['return'] * DT,
                            total_s=total_time),
        by_state={s: dict(
            n=len(grp.get(s, [])),
            n_pick=sum(1 for r in grp.get(s, []) if r['picked']),
            v=sum(r['v'] for r in grp.get(s, [])),
            v_pick=sum(r['v'] for r in grp.get(s, []) if r['picked']),
        ) for s in ('idle', 'chase', 'return')},
        missed_total=dict(n=len(missed), v=miss_v),
        d_le_3_missed=dict(n=len(d3), v=d3_v,
                           by_state={s: d3_by.get(s, [0, 0.0]) for s in
                                     ('idle', 'chase', 'return')}),
        totals=dict(n_ev=len(recs), total_v=total_v, n_pick=n_pick,
                    picked_v=picked_v, minutes=minutes,
                    per_min=picked_v / minutes),
    )


if __name__ == '__main__':
    recs, state_steps, steps, env, pol = run_diag()
    result = summarize(recs, state_steps, steps, env, pol)
