# -*- coding: utf-8 -*-
"""
任务2 机器人拦截策略 · 可移植封装版（单文件，纯 Python 标准库，无任何第三方依赖）
================================================================
接口约定
--------
- 输入：食饵事件序列 [(t, x, y, v), ...]，与题目数据表一致：
    t = 出现时刻（秒）；x, y = 交点坐标（1..10）；v = 分值
- 输出：每秒一个移动增量 (dx, dy)，dx/dy ∈ {-1, 0, 1} 且不同时非零，
    如 (0,0)=原地等待，(0,-1)=向 y 负方向走 1 格。机器人 1m/s，网格间隔 1m，
    故每秒恰好走 1 格或原地不动。

用法
----
1) 嵌入你自己的模拟器（在线逐步调用）：
     from robot_policy import RobotPolicy
     pol = RobotPolicy()                        # 初始位置 (5,5)
     dx, dy = pol.act(t, active_baits)          # active_baits=[(ta,x,y,v),...] 当前在场
2) 直接命令行测试（自带模拟器 + 可选 GUI）：
     python robot_policy.py --file data.csv --start 894
     python robot_policy.py --file data.csv --start 894 --gui
     python robot_policy.py --file data.csv --moves-out moves.csv   # 导出增量序列
   CSV 需含 t,x,y,v 四列（有无表头均可）；装了 pandas 也可直接喂题目 xlsx。

策略设计依据（前期统计分析结论）
--------------------------------
1. 落点不可预测（26 项序列检验 BH 校正后 0 显著）→ 不做落点预测；
   空闲时驻守 WAIT_POINT=(3,7)：经验落点分布下曼哈顿 3m（=存活时间×速度）
   可达概率最大的交点，覆盖约 39.5% 的落点。
2. 分值与位置/间隔独立 → 追击只看当前场面：对所有"赶得上"的食饵做
   带截止期的排列调度（取价值前 5 个全排列，总分最大、并列取完成最早），
   奔向最优方案的第一个目标。
3. 到达节奏可预测（约 94% 在上一枚后 2–10s 内）→ 拾取后立即回撤等待点。
"""
import argparse
import csv
import sys
from itertools import permutations

LIFETIME = 3          # 食饵停留秒数
WAIT_POINT = (3, 7)   # 统计得出的最优驻守交点（表格坐标 1..10）
START_POS = (5, 5)    # 初始位置：场地中心附近交点

# 口径：食饵"停留3秒后消失"，第3秒整（ta+3）机器人恰好踩到算得分。
# 即拾取判定发生在过期判定之前，最晚可在 ta+LIFETIME 到达。
DEADLINE = LIFETIME   # 最晚到达时刻 = ta + DEADLINE


class RobotPolicy:
    """手工启发式策略。act() 每秒调用一次，返回该秒的移动增量。"""

    def __init__(self, start=START_POS):
        self.pos = tuple(start)

    # ---- 内部：带截止期的排列调度，返回最优方案的第一个目标 ----
    def _best_first_target(self, t, active):
        cand = [b for b in active
                if abs(self.pos[0] - b[1]) + abs(self.pos[1] - b[2])
                <= b[0] + DEADLINE - t]
        if not cand:
            return None
        cand = sorted(cand, key=lambda b: -b[3])[:5]  # 最多 5 个，5!=120
        best_v, best_T, best_first = -1, 10 ** 9, None
        for perm in permutations(range(len(cand))):
            px, py, tt, val, first = self.pos[0], self.pos[1], t, 0, None
            for i in perm:
                ta, bx, by, v = cand[i]
                eta = tt + abs(px - bx) + abs(py - by)
                if eta <= ta + DEADLINE:              # 截止期内赶得到才纳入
                    val += v
                    tt, px, py = eta, bx, by
                    if first is None:
                        first = (bx, by)
            if first and (val > best_v or (val == best_v and tt < best_T)):
                best_v, best_T, best_first = val, tt, first
        return best_first

    def act(self, t, active, pos=None):
        """t: 当前时刻（秒）；active: 当前在场食饵 [(ta,x,y,v),...]（表格坐标）。
        返回 (dx, dy)，并同步内部位置。pos 可显式传入以覆盖内部状态。"""
        if pos is not None:
            self.pos = tuple(pos)
        tgt = self._best_first_target(t, active) or WAIT_POINT
        ddx, ddy = tgt[0] - self.pos[0], tgt[1] - self.pos[1]
        if ddx == 0 and ddy == 0:
            return (0, 0)
        if abs(ddx) >= abs(ddy):                      # 先走差距大的轴
            step = (1 if ddx > 0 else -1, 0)
        else:
            step = (0, 1 if ddy > 0 else -1)
        self.pos = (self.pos[0] + step[0], self.pos[1] + step[1])
        return step


# ================= 自带模拟器（测试/评估用） =================
def simulate(events, start_idx=0, collect_moves=False, on_step=None):
    """从第 start_idx 条（0-based）投放回放到结束。首枚移到 t=3s 给起步时间。
    口径：第3秒整踩到算得分（拾取判定先于过期）。
    返回 dict(score, picked, seconds, per_min, moves)。"""
    ev = sorted(events, key=lambda e: e[0])[start_idx:]
    if not ev:
        return dict(score=0, picked=0, seconds=0, per_min=0.0, moves=[])
    shift = ev[0][0] - 3
    ev = [(e[0] - shift, e[1], e[2], e[3]) for e in ev]

    pol = RobotPolicy()
    pos = pol.pos
    active, nxt = [], 0
    t, score, picked = 0, 0, 0
    moves = []
    while nxt < len(ev) or active:
        while nxt < len(ev) and ev[nxt][0] <= t:      # 投放
            active.append(ev[nxt]); nxt += 1
        for b in [b for b in active if (b[1], b[2]) == pos]:  # 拾取（先于过期：第3秒边界算拾到）
            score += b[3]; picked += 1
            active.remove(b)
        active = [b for b in active if t - b[0] < LIFETIME]   # 过期
        if on_step and on_step(t, pos, active, score, picked) is False:
            break
        dx, dy = pol.act(t, active, pos=pos)
        if collect_moves:
            moves.append((t, dx, dy))
        pos = (pos[0] + dx, pos[1] + dy)
        assert 1 <= pos[0] <= 10 and 1 <= pos[1] <= 10, f'越界 {pos}'
        t += 1
    minutes = t / 60
    return dict(score=score, picked=picked, seconds=t,
                per_min=score / minutes if minutes else 0.0, moves=moves)


# ================= 数据读取 =================
def load_events(path):
    """CSV（t,x,y,v 四列，有无表头均可）；.xlsx 需已安装 pandas。
    兼容题目附件格式：第2列为 "(x,y)" 字符串时自动解析。"""
    rows = []
    if path.lower().endswith(('.xlsx', '.xls')):
        import pandas as pd
        raw = pd.read_excel(path).values.tolist()
    else:
        with open(path, newline='', encoding='utf-8-sig') as f:
            raw = list(csv.reader(f))
    for r in raw:
        r = [c for c in r if str(c).strip() != '']
        if not r:
            continue
        try:
            if len(r) >= 4:
                t, x, y, v = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            else:                                     # (t, "(x,y)", v) 题目附件格式
                sx, sy = str(r[1]).strip().strip('()').split(',')
                t, x, y, v = float(r[0]), float(sx), float(sy), float(r[2])
        except (ValueError, IndexError):
            continue                                  # 跳过表头/坏行
        assert 1 <= x <= 10 and 1 <= y <= 10, f'坐标应在 1..10: {r}'
        rows.append((t, int(x), int(y), float(v)))
    if not rows:
        raise ValueError('未解析到任何事件，请确认列格式为 t,x,y,v')
    return rows


# ================= 可选 GUI（tkinter，标准库） =================
def run_gui(events, start_idx):
    import tkinter as tk
    from tkinter import ttk
    CELL, PAD = 52, 40
    W = 9 * CELL + PAD * 2

    root = tk.Tk()
    root.title('机器人拦截策略回放（手工算法·封装版）')
    top = ttk.Frame(root); top.pack(fill='x', padx=8, pady=4)
    stat = ttk.Label(top, text='', font=('Microsoft YaHei', 11)); stat.pack(side='left')
    speed = tk.DoubleVar(value=8.0)
    ttk.Scale(top, from_=1, to=60, variable=speed, length=160).pack(side='right')
    ttk.Label(top, text='倍速').pack(side='right')
    cv = tk.Canvas(root, width=W, height=W, bg='#fafaf5', highlightthickness=0)
    cv.pack(padx=8, pady=6)

    def px(x, y):                                     # 表格坐标 1..10 → 像素
        return PAD + (x - 1) * CELL, PAD + (10 - y) * CELL

    frames = []
    simulate(events, start_idx,
             on_step=lambda t, pos, active, score, picked:
             frames.append((t, pos, [tuple(b) for b in active], score, picked)))

    state = {'i': 0}

    def draw():
        if state['i'] >= len(frames):
            stat.config(text=stat.cget('text') + '  ―― 播放完毕')
            return
        t, pos, active, score, picked = frames[state['i']]
        state['i'] += 1
        cv.delete('all')
        for i in range(10):
            x0, y0 = px(i + 1, 1); x1, y1 = px(i + 1, 10)
            cv.create_line(x0, y0, x1, y1, fill='#c8c8c0')
            x0, y0 = px(1, i + 1); x1, y1 = px(10, i + 1)
            cv.create_line(x0, y0, x1, y1, fill='#c8c8c0')
        wx, wy = px(*WAIT_POINT)
        cv.create_oval(wx - 12, wy - 12, wx + 12, wy + 12, outline='#8888dd',
                       width=2, dash=(3, 2))
        for ta, bx, by, v in active:
            frac = max((LIFETIME - (t - ta)) / LIFETIME, 0)
            x, y = px(bx, by)
            col = '#2e8b40' if frac > 0.55 else ('#d99a00' if frac > 0.25 else '#cc3333')
            r0 = 6 + 7 * frac
            cv.create_oval(x - r0, y - r0, x + r0, y + r0, outline=col, width=3)
            cv.create_text(x, y, text=str(int(v)), fill=col,
                           font=('Microsoft YaHei', 10, 'bold'))
        x, y = px(*pos)
        cv.create_oval(x - 8, y - 8, x + 8, y + 8, fill='#c62828', outline='#7a1010')
        per_min = score / t * 60 if t else 0
        stat.config(text=f't={t}s | 得分 {score} | 拾取 {picked} | {per_min:.1f} 分/分钟')
        root.after(max(int(1000 / speed.get()), 5), draw)

    draw()
    root.mainloop()


# ================= 命令行入口 =================
def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description='机器人拦截策略（手工算法封装版）')
    ap.add_argument('--file', required=True, help='事件序列 CSV/xlsx（t,x,y,v，坐标1..10）')
    ap.add_argument('--start', type=int, default=1, help='起始投放序号（1-based，默认1）')
    ap.add_argument('--moves-out', default=None, help='把每秒增量 (t,dx,dy) 写出到 CSV')
    ap.add_argument('--gui', action='store_true', help='tkinter 回放动画')
    args = ap.parse_args()

    events = load_events(args.file)
    print(f'事件 {len(events)} 条，从第 {args.start} 条开始模拟')
    if args.gui:
        run_gui(events, args.start - 1)
        return
    r = simulate(events, args.start - 1, collect_moves=bool(args.moves_out))
    print(f'总分 {r["score"]:.0f} | 拾取 {r["picked"]} 个 | '
          f'时长 {r["seconds"]}s ({r["seconds"]/60:.1f}min) | '
          f'{r["per_min"]:.2f} 分/分钟')
    if args.moves_out:
        with open(args.moves_out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['t', 'dx', 'dy'])
            w.writerows(r['moves'])
        print(f'增量序列已写出: {args.moves_out}（{len(r["moves"])} 行）')


if __name__ == '__main__':
    main()
