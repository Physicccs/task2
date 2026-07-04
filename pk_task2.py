# -*- coding: utf-8 -*-
"""
任务2 统一接口赛马场
====================================
唯一模型形式 = 统一赛马接口：.py 文件内含 ``RobotPolicy`` 类，
    act(self, t, active, pos=None) → (dx, dy)
    - t: 当前秒（整数）
    - active: 当前在场食饵 [(出现时刻, x, y, v), ...]，表格坐标 1..10
    - pos: 机器人当前位置 (x, y)
    - 返回 (0,0)/(±1,0)/(0,±1)，每秒调用一次（机器人 1m/s，每秒走一格）
模板 = task2/robot_policy.py；合规示例 = task2/heuristic_v2.py、task2/ext_ml1.py、
算法库/your_model.py。

无默认模型：命令行用 --policy 指定（可多次）；GUI 用「添加模型…」按钮自选。
每局评测都新建策略实例，模型内部状态天然不跨局。

口径：每秒一格、位置相等即得分、第 3 秒整踩到算得分（拾取判定先于过期）。
数据：默认题目附件；--file 可换 CSV/xlsx（t,x,y,v 四列，或题目附件 (t,"(x,y)",v)）。

合成数据模式（防过拟合 / 稳健性测试）：--synthetic MIN 按任务1拟合规律
（间隔~离散正态、位置~经验多项分布、分值~奇偶增强截断几何，见 bait_gen.py）
生成 MIN 分钟全新序列 × --reps 局评测，报均值±σ。测试集是新生成的，
真实附件全部 1787 条可放心用于模型调参。GUI 里对应「合成序列」按钮。

用法（从项目根目录运行）:
  D:/conda/envs/AI/python.exe task2/pk_task2.py --policy task2/heuristic_v2.py
  D:/conda/envs/AI/python.exe task2/pk_task2.py --policy a.py --policy b.py --start 894
  D:/conda/envs/AI/python.exe task2/pk_task2.py --policy a.py --synthetic 90 --reps 10
  D:/conda/envs/AI/python.exe task2/pk_task2.py --gui          # 空场启动，GUI 里选模型
"""
import argparse
import importlib.util
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_policy import load_events, LIFETIME        # noqa: E402  零依赖数据解析
import bait_gen                                       # noqa: E402  合成序列（任务1拟合规律）

INPUT = 'input/202607032000建模复赛题附件.xlsx'
GRID = 10                 # 表格坐标 1..10
START_POS = (5, 5)
VAL_FRAC = 0.8            # 后 20%（按时间）为独立验证段

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ---------------- 模型加载（统一接口） ----------------
def load_policy_file(path):
    """加载 .py 模型文件，要求含 RobotPolicy 类。返回 (名字, 策略类)。"""
    if not path.lower().endswith('.py'):
        raise ValueError(f'{path}: 只支持 .py 统一接口模型（需含 RobotPolicy 类）')
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f'ext_{name}', os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'RobotPolicy'):
        raise ValueError(f'{path} 中未找到 RobotPolicy 类（统一接口要求）')
    return name, mod.RobotPolicy


def _sanitize(step):
    """容错清洗模型输出：取整、限幅 ±1、双轴同时非零时保留 x 轴。"""
    try:
        dx, dy = step
        dx, dy = int(round(dx)), int(round(dy))
    except (TypeError, ValueError):
        return 0, 0
    dx = max(-1, min(1, dx))
    dy = max(-1, min(1, dy))
    if dx != 0 and dy != 0:
        dy = 0
    return dx, dy


# ---------------- 整秒模拟内核 ----------------
def shift_events(events, start_idx):
    """第 start_idx(0-based) 条起回放，首枚移到 t=3 给起步时间。"""
    ev = sorted(events, key=lambda e: e[0])[start_idx:]
    if not ev:
        return []
    shift = ev[0][0] - 3
    return [(e[0] - shift, e[1], e[2], e[3]) for e in ev]


class Arena:
    """单模型一局模拟的状态机；step() 推进 1 秒。GUI 与命令行共用。"""

    def __init__(self, events, pol_cls, start_idx=0):
        self.ev = shift_events(events, start_idx)
        self.pol = pol_cls()                    # 每局全新实例，无跨局状态
        self.pos = START_POS
        self.active, self.nxt = [], 0
        self.t, self.score, self.picked = 0, 0.0, 0
        self.err = None                         # 模型运行时异常（GUI 显示用）
        self.just_picked = []                   # 本秒拾到的 (x, y, v)

    def finished(self):
        return self.nxt >= len(self.ev) and not self.active

    def step(self):
        if self.finished():
            return False
        t = self.t
        while self.nxt < len(self.ev) and self.ev[self.nxt][0] <= t:   # 投放
            self.active.append(self.ev[self.nxt])
            self.nxt += 1
        self.just_picked = []
        for b in [b for b in self.active if (b[1], b[2]) == self.pos]:  # 拾取（先于过期）
            self.score += b[3]
            self.picked += 1
            self.just_picked.append((b[1], b[2], b[3]))
            self.active.remove(b)
        self.active = [b for b in self.active if t - b[0] < LIFETIME]   # 过期
        dx = dy = 0
        if self.err is None:
            try:
                dx, dy = _sanitize(self.pol.act(t, list(self.active), pos=self.pos))
            except Exception as ex:             # 模型异常 → 原地罚站并记录，不拖垮全场
                self.err = f'{type(ex).__name__}: {ex}'
        self.pos = (max(1, min(GRID, self.pos[0] + dx)),
                    max(1, min(GRID, self.pos[1] + dy)))
        self.t += 1
        return not self.finished()


def simulate(events, pol_cls, start_idx=0):
    a = Arena(events, pol_cls, start_idx)
    while a.step():
        pass
    minutes = a.t / 60.0
    return dict(score=a.score, picked=a.picked, minutes=minutes,
                per_min=a.score / minutes if minutes else 0.0, err=a.err)


# ---------------- 命令行评估 ----------------
def run_eval(events, start_idx, policies):
    ev_sorted = sorted(events, key=lambda e: e[0])
    t_cut = ev_sorted[-1][0] * VAL_FRAC
    val_idx = sum(1 for e in ev_sorted if e[0] <= t_cut)

    segs = [(f'指定段（第 {start_idx + 1} 条 → 结束，共 {len(events) - start_idx} 条）',
             start_idx)]
    if val_idx != start_idx and val_idx < len(events):
        segs.append((f'后20%验证段（第 {val_idx + 1} 条 → 结束，共 {len(events) - val_idx} 条）',
                     val_idx))

    for title, idx in segs:
        print(f'\n== {title} ==')
        print(f'{"策略":<14}{"总分":>10}{"拾取数":>8}{"时长(min)":>12}{"分/分钟":>10}')
        for name, cls in policies.items():
            r = simulate(events, cls, idx)
            tag = f'  [异常: {r["err"]}]' if r['err'] else ''
            print(f'{name:<14}{r["score"]:>10.0f}{r["picked"]:>8d}'
                  f'{r["minutes"]:>12.1f}{r["per_min"]:>10.2f}{tag}')


def run_eval_synthetic(real_events, policies, minutes, seed0, reps):
    """合成序列稳健性测试：按任务1拟合规律生成 reps 局全新序列，报均值±σ。

    real_events 仅用于位置经验分布 π̂；间隔/分值按拟合参数采样，
    测试集与题目附件无一枚重合 → 模型可放开在全部真实数据上调参。
    """
    print(f'\n== 合成序列稳健性测试（任务1拟合规律生成 · {minutes:g} 分钟 × {reps} 局 · '
          f'seed {seed0}–{seed0 + reps - 1}）==')
    results = {n: [] for n in policies}
    for r in range(reps):
        ev = bait_gen.generate(minutes, seed0 + r, real_events)
        line = [f'seed {seed0 + r} ({len(ev)} 枚):']
        for name, cls in policies.items():
            res = simulate(ev, cls, 0)
            results[name].append(res)
            line.append(f'{name} {res["per_min"]:.2f}' + ('!' if res['err'] else ''))
        print('  ' + '  |  '.join(line))
    print(f'\n{"策略":<14}{"分/分钟 μ":>11}{"σ":>7}{"最差局":>9}{"最好局":>9}{"拾取/局":>9}')
    for name, rs in results.items():
        pm = [x['per_min'] for x in rs]
        mean = sum(pm) / len(pm)
        std = (sum((v - mean) ** 2 for v in pm) / (len(pm) - 1)) ** 0.5 if len(pm) > 1 else 0.0
        err = next((x['err'] for x in rs if x['err']), None)
        tag = f'  [异常: {err}]' if err else ''
        print(f'{name:<14}{mean:>11.2f}{std:>7.2f}{min(pm):>9.2f}{max(pm):>9.2f}'
              f'{sum(x["picked"] for x in rs) / len(rs):>9.1f}{tag}')


# ---------------- 多画布 GUI ----------------
CELL = 46
PAD = 34
W = (GRID - 1) * CELL + PAD * 2


class Panel:
    """单模型画布：持有自己的 Arena / 轨迹 / 闪光，负责绘制。"""

    def __init__(self, parent, title, pol_cls, tk, ttk):
        self.tk = tk
        self.pol_cls = pol_cls
        self.frame = ttk.Frame(parent)
        self.frame.pack(side='left', padx=6)
        self.head = ttk.Label(self.frame, text=title, font=('Microsoft YaHei', 12, 'bold'))
        self.head.pack()
        self.cv = tk.Canvas(self.frame, width=W, height=W, bg='#fafaf5',
                            highlightthickness=0)
        self.cv.pack()
        self.stat = ttk.Label(self.frame, text='', font=('Microsoft YaHei', 10))
        self.stat.pack()
        self.arena = None
        self.trail = []
        self.flashes = []

    def px(self, x, y):
        return PAD + (x - 1) * CELL, PAD + (GRID - y) * CELL

    def start(self, events, start_idx):
        self.arena = Arena(events, self.pol_cls, start_idx)
        self.trail = [self.arena.pos]
        self.flashes = []

    def step(self):
        a = self.arena
        if a is None or a.finished():
            return False
        alive = a.step()
        for (x, y, v) in a.just_picked:
            self.flashes.append([x, y, v, 0])
        self.trail.append(a.pos)
        if len(self.trail) > 30:
            self.trail.pop(0)
        return alive

    def draw(self):
        cv = self.cv
        cv.delete('all')
        for i in range(1, GRID + 1):
            x0, y0 = self.px(i, 1); x1, y1 = self.px(i, GRID)
            cv.create_line(x0, y0, x1, y1, fill='#c8c8c0')
            x0, y0 = self.px(1, i); x1, y1 = self.px(GRID, i)
            cv.create_line(x0, y0, x1, y1, fill='#c8c8c0')
        a = self.arena
        if a is None:
            return
        if len(self.trail) > 1:
            pts = [c for p in self.trail for c in self.px(*p)]
            cv.create_line(*pts, fill='#e08080', width=2)
        t_now = a.t - 1                       # 画面对应"刚推进完"的那一秒
        for b in a.active:
            frac = max((LIFETIME - (t_now - b[0])) / LIFETIME, 0)
            x, y = self.px(b[1], b[2])
            r0 = 5 + 6 * frac
            col = '#2e8b40' if frac > 0.55 else ('#d99a00' if frac > 0.25 else '#cc3333')
            cv.create_oval(x - r0, y - r0, x + r0, y + r0, outline=col, width=3)
            cv.create_text(x, y, text=f'{int(b[3])}', fill=col,
                           font=('Microsoft YaHei', 9, 'bold'))
        for f in list(self.flashes):
            f[3] += 1
            if f[3] > 10:
                self.flashes.remove(f)
                continue
            x, y = self.px(f[0], f[1])
            cv.create_text(x, y - 12 - f[3] * 2, text=f'+{int(f[2])}',
                           fill='#cc7700', font=('Microsoft YaHei', 11, 'bold'))
        x, y = self.px(*a.pos)
        cv.create_oval(x - 7, y - 7, x + 7, y + 7, fill='#c62828',
                       outline='#7a1010', width=2)
        per_min = a.score / a.t * 60 if a.t > 1 else 0
        txt = (f'得分 {a.score:.0f} | 拾取 {a.picked} | {per_min:.1f} 分/分钟 | '
               f'投放 {a.nxt}/{len(a.ev)}')
        if a.err:
            txt += f'\n模型异常: {a.err}'
        self.stat.config(text=txt)

    def destroy(self):
        self.frame.destroy()


def run_gui(events, start_default, policies):
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    root = tk.Tk()
    root.title('任务2 赛马场 · 统一接口 RobotPolicy · 同序列同步回放')

    data = {'ev': events, 'label': '题目附件'}      # 当前回放数据源（可切合成序列）

    bar = ttk.Frame(root); bar.pack(fill='x', padx=8, pady=(6, 2))
    start_lbl = ttk.Label(bar, text=f'从第几次投放开始 (1–{len(events)}):')
    start_lbl.pack(side='left')
    idx_var = tk.StringVar(value=str(start_default))
    ttk.Entry(bar, textvariable=idx_var, width=7).pack(side='left', padx=4)
    btn = ttk.Button(bar, text='开始'); btn.pack(side='left', padx=4)
    ttk.Button(bar, text='重置', command=lambda: reset()).pack(side='left', padx=4)
    ttk.Button(bar, text='添加模型…', command=lambda: add_model()).pack(side='left', padx=4)
    ttk.Label(bar, text='倍速:').pack(side='left', padx=(12, 2))
    speed = tk.DoubleVar(value=4.0)
    ttk.Scale(bar, from_=0.5, to=30, variable=speed, length=140).pack(side='left')
    spd_lbl = ttk.Label(bar, text='4.0×'); spd_lbl.pack(side='left', padx=2)
    clock = ttk.Label(bar, text='', font=('Consolas', 11)); clock.pack(side='right')

    # 数据源栏：真实附件 ↔ 合成序列（任务1拟合规律生成，防过拟合测试）
    bar2 = ttk.Frame(root); bar2.pack(fill='x', padx=8, pady=(0, 4))
    src_lbl = ttk.Label(bar2, text=f'数据源: 题目附件（{len(events)} 枚）',
                        font=('Microsoft YaHei', 9))
    src_lbl.pack(side='left')
    ttk.Label(bar2, text='   合成时长(min):').pack(side='left')
    syn_min_var = tk.StringVar(value='90')
    ttk.Entry(bar2, textvariable=syn_min_var, width=6).pack(side='left', padx=2)
    ttk.Label(bar2, text='种子(空=随机):').pack(side='left', padx=(6, 0))
    syn_seed_var = tk.StringVar(value='')
    ttk.Entry(bar2, textvariable=syn_seed_var, width=8).pack(side='left', padx=2)
    ttk.Button(bar2, text='合成序列',
               command=lambda: new_synthetic()).pack(side='left', padx=4)
    ttk.Button(bar2, text='真实数据',
               command=lambda: use_real()).pack(side='left', padx=2)

    body = ttk.Frame(root); body.pack(padx=8, pady=4)
    panels = [Panel(body, n, cls, tk, ttk) for n, cls in policies.items()]
    if not panels:
        hint = ttk.Label(body, text='尚未加载模型\n\n点击上方「添加模型…」\n'
                                    '选择含 RobotPolicy 类的 .py 文件',
                         font=('Microsoft YaHei', 12), justify='center')
        hint.pack(padx=60, pady=80)
    else:
        hint = None

    state = {'running': False}

    def _set_source(ev, label):
        data['ev'] = ev
        data['label'] = label
        start_lbl.config(text=f'从第几次投放开始 (1–{len(ev)}):')
        src_lbl.config(text=f'数据源: {label}（{len(ev)} 枚）')
        reset()

    def new_synthetic():
        try:
            minutes = float(syn_min_var.get())
            assert minutes > 0
        except (ValueError, AssertionError):
            messagebox.showerror('输入错误', '合成时长请输入正数（分钟）')
            return
        txt = syn_seed_var.get().strip()
        try:
            seed = int(txt) if txt else random.randrange(1_000_000)
        except ValueError:
            messagebox.showerror('输入错误', '种子请输入整数，或留空随机')
            return
        syn_seed_var.set(str(seed))            # 回显种子，保证可复现
        idx_var.set('1')
        _set_source(bait_gen.generate(minutes, seed, events),
                    f'合成 seed={seed}·{minutes:g}min')

    def use_real():
        idx_var.set(str(start_default))
        _set_source(events, '题目附件')

    def add_model():
        nonlocal hint
        path = filedialog.askopenfilename(
            title='选择模型文件（统一接口：.py 含 RobotPolicy 类）',
            filetypes=[('RobotPolicy 模型', '*.py'), ('所有文件', '*.*')])
        if not path:
            return
        try:
            name, cls = load_policy_file(path)
            cls()                                # 提前实例化一次，暴露加载期错误
        except Exception as ex:
            messagebox.showerror('加载失败', str(ex))
            return
        names = {p.head.cget('text') for p in panels}
        while name in names:
            name += '_2'
        if hint is not None:
            hint.destroy()
            hint = None
        panels.append(Panel(body, name, cls, tk, ttk))
        reset()

    def reset():
        state['running'] = False
        btn.config(text='开始')
        for p in panels:
            p.arena = None
            p.cv.delete('all')
            p.stat.config(text='')
        clock.config(text='')

    def toggle():
        if state['running']:
            state['running'] = False
            btn.config(text='继续')
            return
        if not panels:
            messagebox.showinfo('无模型', '请先点「添加模型…」选择 .py 模型文件')
            return
        if panels[0].arena is None:
            ev = data['ev']
            try:
                idx = int(idx_var.get()) - 1
                assert 0 <= idx < len(ev)
            except (ValueError, AssertionError):
                messagebox.showerror('输入错误', f'请输入 1–{len(ev)} 的整数')
                return
            for p in panels:
                p.start(ev, idx)
        state['running'] = True
        btn.config(text='暂停')
        tick()

    def tick():
        if not state['running']:
            return
        alive = False
        for p in panels:
            alive = p.step() or alive
            p.draw()
        clock.config(text=f't = {panels[0].arena.t:6d}s')
        sp = speed.get()
        spd_lbl.config(text=f'{sp:.1f}×')
        if not alive:
            state['running'] = False
            btn.config(text='开始')
            lead = max(panels, key=lambda p: p.arena.score)
            clock.config(text=f'播放完毕，胜者: {lead.head.cget("text")} '
                              f'({lead.arena.score:.0f} 分)')
            return
        root.after(max(int(1000 / sp), 5), tick)

    btn.config(command=toggle)
    root.mainloop()


# ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='任务2 统一接口赛马场')
    ap.add_argument('--start', type=int, default=894, help='起始投放序号(1-based)')
    ap.add_argument('--file', default=None, help='自定义事件序列 CSV/xlsx (t,x,y,v)')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--policy', action='append', default=[],
                    help='模型 .py 文件（统一接口，需含 RobotPolicy 类；可多次）')
    ap.add_argument('--synthetic', type=float, default=None, metavar='MIN',
                    help='合成数据模式：按任务1拟合规律生成 MIN 分钟全新序列做测试集'
                         '（真实数据可全量用于调参）')
    ap.add_argument('--seed', type=int, default=2026, help='合成序列起始种子')
    ap.add_argument('--reps', type=int, default=5,
                    help='合成模式局数（种子逐局 +1），报均值±σ（默认 5）')
    args = ap.parse_args()

    policies = {}
    for p in args.policy:
        name, cls = load_policy_file(p)
        while name in policies:
            name += '_2'
        policies[name] = cls

    events = load_events(args.file or INPUT)
    ts = [e[0] for e in events]
    print(f'事件总数 {len(events)}，时间跨度 {min(ts):.0f}–{max(ts):.0f}s')

    if args.gui:
        run_gui(events, args.start, policies)
    elif not policies:
        print('\n未指定模型。用 --policy 加载统一接口模型（可多次），例如：\n'
              '  D:/conda/envs/AI/python.exe task2/pk_task2.py '
              '--policy task2/heuristic_v2.py --policy 算法库/your_model.py\n'
              '加 --synthetic 90 用任务1拟合规律生成的全新序列测试（防过拟合），\n'
              '或加 --gui 在图形界面里选择模型。')
        sys.exit(1)
    elif args.synthetic:
        run_eval_synthetic(events, policies, args.synthetic, args.seed, args.reps)
    else:
        run_eval(events, args.start - 1, policies)


if __name__ == '__main__':
    main()
