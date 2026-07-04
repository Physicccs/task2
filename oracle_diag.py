# -*- coding: utf-8 -*-
"""
任务2 诊断与理论上界
====================================
目的：回答"手工策略还能不能更强"，给出可量化的标尺。

三条基准（整秒模型，与 robot_policy.py 封装版口径一致：第3秒整踩到算得分）：
1. 硬上界   = 所有食饵总价值（瞬移拿满，物理不可能，只作参照）
2. Oracle   = 全知未来 + beam search 近似最优路径（实用天花板）
3. 当前策略 = RobotPolicy（robot_policy.py）实际得分

再对当前策略做逐枚食饵损失归因：
- picked            : 拾到
- soft_missed       : 存活期内机器人曾"直达可及"却没拾（站位/调度可救 → 改进空间）
- hard_missed       : 从机器人当时轨迹永远够不到（受前序决策牵连）
- oracle_gain       : oracle 拾到但策略没拾（真正的可争取分）

用法: D:/conda/envs/AI/python.exe task2/oracle_diag.py --start 894 [--beam 2000]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from robot_policy import RobotPolicy, load_events, simulate, LIFETIME, START_POS

INPUT = 'input/202607032000建模复赛题附件.xlsx'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def shift_events(events, start_idx):
    """从第 start_idx(0-based) 条起，首枚移到 t=3，返回整数时刻事件列表（升序）。"""
    ev = sorted(events, key=lambda e: e[0])[start_idx:]
    if not ev:
        return []
    shift = ev[0][0] - 3
    return [(int(round(e[0] - shift)), int(e[1]), int(e[2]), float(e[3])) for e in ev]


# ---------------- Oracle: 全知 beam search ----------------
def oracle_beam(events, beam=2000, start=START_POS):
    """整秒模型下的近似最优（全知未来）。返回 (best_score, best_picked_ids)。

    状态 = (pos, frozenset(已拾且仍在窗口内的食饵id))；逐秒推进，每步 5 个动作
    （停/上/下/左/右），到达坐标即拾取该点未拾食饵。beam 宽度剪枝。
    """
    if not events:
        return 0.0, set()
    T = max(e[0] for e in events) + LIFETIME + 1
    # 每个整秒的在场食饵 id：ta <= t <= ta+LIFETIME（第3秒可拾）
    active_at = {t: [] for t in range(T + 1)}
    for i, (ta, x, y, v) in enumerate(events):
        for t in range(ta, min(ta + int(LIFETIME), T) + 1):
            active_at[t].append(i)
    dead = [events[i][0] + int(LIFETIME) for i in range(len(events))]

    # state key=(pos, frozenset picked_recent) -> (score, picked_full_frozenset)
    init = (tuple(start), frozenset())
    states = {init: (0.0, frozenset())}
    moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    for t in range(T + 1):
        act_ids = active_at.get(t, [])
        nxt = {}
        for (pos, recent), (sc, full) in states.items():
            for dx, dy in moves:
                nx, ny = pos[0] + dx, pos[1] + dy
                if not (1 <= nx <= 10 and 1 <= ny <= 10):
                    continue
                gain = 0.0
                got = []
                for i in act_ids:
                    if i in full:
                        continue
                    if events[i][1] == nx and events[i][2] == ny:
                        gain += events[i][3]
                        got.append(i)
                nsc = sc + gain
                nfull = full | frozenset(got) if got else full
                # recent 只保留仍在窗口的，控制 key 规模
                nrecent = frozenset(i for i in (recent | frozenset(got)) if dead[i] >= t)
                key = ((nx, ny), nrecent)
                cur = nxt.get(key)
                if cur is None or nsc > cur[0]:
                    nxt[key] = (nsc, nfull)
        if len(nxt) > beam:
            top = sorted(nxt.items(), key=lambda kv: -kv[1][0])[:beam]
            nxt = dict(top)
        states = nxt

    best_key = max(states, key=lambda k: states[k][0])
    best_sc, best_full = states[best_key]
    return best_sc, set(best_full)


# ---------------- 当前策略损失归因 ----------------
def diagnose_policy(events, start=START_POS):
    """跑 RobotPolicy，记录每秒轨迹，对每枚食饵归因。"""
    # 记录每秒 (t, pos, active_ids)
    trace = []
    idx_by_key = {(e[0], e[1], e[2], e[3]): i for i, e in enumerate(events)}
    ev_sorted = events

    def on_step(t, pos, active, score, picked):
        aid = []
        for b in active:
            k = (b[0], b[1], b[2], b[3])
            if k in idx_by_key:
                aid.append(idx_by_key[k])
        trace.append((t, tuple(pos), aid))
        return True

    r = simulate([(e[0], e[1], e[2], e[3]) for e in ev_sorted], 0,
                 collect_moves=False, on_step=on_step)

    # 哪些被拾取：用 picked 数与轨迹重建。simulate 内部按 pos 命中拾取，
    # 这里用"食饵坐标在某秒 == 机器人位置且在存活窗口"判定拾取归属。
    picked_ids = set()
    pos_at = {t: p for (t, p, _) in trace}
    for i, (ta, x, y, v) in enumerate(events):
        for t in range(ta, ta + int(LIFETIME) + 1):
            if pos_at.get(t) == (x, y):
                picked_ids.add(i)
                break

    # soft/hard missed
    soft, hard = set(), set()
    for i, (ta, x, y, v) in enumerate(events):
        if i in picked_ids:
            continue
        reachable_sometime = False
        for t in range(ta, ta + int(LIFETIME) + 1):
            p = pos_at.get(t)
            if p is None:
                continue
            d = abs(p[0] - x) + abs(p[1] - y)
            if d <= (ta + int(LIFETIME) - t):     # 从当时位置直奔能在 deadline 前到
                reachable_sometime = True
                break
        (soft if reachable_sometime else hard).add(i)

    return r, picked_ids, soft, hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=894)
    ap.add_argument('--beam', type=int, default=2000)
    ap.add_argument('--file', default=INPUT)
    args = ap.parse_args()

    raw = load_events(args.file)
    ev = shift_events(raw, args.start - 1)
    total_v = sum(e[3] for e in ev)
    minutes = (max(e[0] for e in ev) + LIFETIME) / 60.0
    print(f'事件 {len(ev)} 条（第 {args.start} 条起），总价值 {total_v:.0f}，'
          f'时长 ~{minutes:.1f} min')

    # 当前策略
    r, picked, soft, hard = diagnose_policy(ev)
    print(f'\n【当前策略 RobotPolicy】')
    print(f'  得分 {r["score"]:.0f} | 拾取 {r["picked"]} | {r["per_min"]:.2f} 分/分钟')
    vp = sum(ev[i][3] for i in picked)
    vs = sum(ev[i][3] for i in soft)
    vh = sum(ev[i][3] for i in hard)
    print(f'  拾到       {len(picked):4d} 枚 / {vp:6.0f} 分')
    print(f'  软错过     {len(soft):4d} 枚 / {vs:6.0f} 分  ← 曾直达可及却没拾（站位/调度可救）')
    print(f'  硬错过     {len(hard):4d} 枚 / {vh:6.0f} 分  ← 轨迹够不到（受前序决策牵连）')

    # Oracle
    print(f'\n【Oracle 全知 beam={args.beam}】计算中…')
    osc, oids = oracle_beam(ev, beam=args.beam)
    print(f'  得分 {osc:.0f} | 拾取 {len(oids)} | {osc/minutes:.2f} 分/分钟')
    gain_ids = oids - picked
    print(f'  oracle 拾到但策略没拾: {len(gain_ids)} 枚 / '
          f'{sum(ev[i][3] for i in gain_ids):.0f} 分')

    print(f'\n【标尺】')
    print(f'  硬上界(总价值)   {total_v:8.0f}   {total_v/minutes:6.2f} 分/分钟')
    print(f'  Oracle(全知最优) {osc:8.0f}   {osc/minutes:6.2f} 分/分钟   '
          f'(达成率 {osc/total_v*100:.1f}%)')
    print(f'  当前策略         {r["score"]:8.0f}   {r["per_min"]:6.2f} 分/分钟   '
          f'(oracle 的 {r["score"]/osc*100:.1f}%)')
    print(f'\n  与 oracle 的差距 = {osc - r["score"]:.0f} 分 '
          f'（{(osc-r["score"])/minutes:.2f} 分/分钟 可争取）')


if __name__ == '__main__':
    main()
