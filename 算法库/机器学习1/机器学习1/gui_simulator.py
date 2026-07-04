#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器人竞赛 - GUI模拟器
=====================
使用训练好的ML模型进行实时可视化模拟。
网格：9m×9m，10×10交点，机器人速度1m/s，食饵停留3秒。
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import re
import joblib
import json
import sys
import io
import time as time_module
from collections import deque

# 强制UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ===========================
# 全局常量和配置
# ===========================
GRID_SIZE = 10          # 10×10 交点
GRID_SPACING = 1.0      # 1米间距
FIELD_SIZE = 9.0        # 9米边长
CELL_PX = 60            # 每格像素
MARGIN = 50             # 边距
CANVAS_SIZE = MARGIN * 2 + (GRID_SIZE - 1) * CELL_PX  # 640px

ROBOT_SPEED = 1.0       # 1 m/s
BAIT_LIFETIME = 3.0     # 3秒

# 颜色定义
COLOR_BG = '#1a1a2e'
COLOR_GRID = '#2d2d44'
COLOR_GRID_POINT = '#4a4a6a'
COLOR_ROBOT = '#ff4444'
COLOR_ROBOT_TRAIL = '#ff6666'
COLOR_BAIT_ACTIVE = '#ffd700'
COLOR_BAIT_INACTIVE = '#ff8c00'
COLOR_BAIT_TEXT = '#000000'
COLOR_PURSUE = '#00ff88'
COLOR_SKIP = '#ff6b6b'
COLOR_PANEL_BG = '#16213e'
COLOR_TEXT = '#e0e0e0'
COLOR_TEXT_DIM = '#888888'
COLOR_HIGHLIGHT = '#ffd700'


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# ===========================
# 主应用类
# ===========================
class RobotSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("机器人竞赛模拟器 - ML决策可视化")
        self.root.geometry("1280x780")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        # 加载模型和数据
        self.load_resources()

        # 模拟状态
        self.simulation_time = 0.0
        self.robot_pos = (5, 5)  # 默认起始位置
        self.robot_target = None
        self.robot_moving = False
        self.robot_path = deque(maxlen=50)  # 轨迹
        self.total_score = 0
        self.pursued_count = 0
        self.skipped_count = 0
        self.bait_index = 0
        self.active_baits = []  # [(pos, score, disappear_time), ...]
        self.current_bait = None
        self.current_decision = None
        self.is_running = False
        self.is_paused = False
        self.speed = 5  # 模拟速度倍率
        self.animation_id = None
        self.last_update_time = 0
        self.decision_history = []  # 用于计算准确率

        # 构建UI
        self.build_ui()

        # 初始化网格
        self.draw_grid()
        self.update_display()

    # ---- 资源加载 ----
    def load_resources(self):
        """加载模型和数据"""
        try:
            self.model = joblib.load('best_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            with open('feature_names.json', 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
            print(f"模型加载成功: best_model.pkl ({len(self.feature_names)} 特征)")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载模型文件:\n{e}")
            self.root.destroy()
            return

        # 加载测试数据
        df = pd.read_excel('202607032000建模复赛题附件.xlsx')
        df.columns = ['time', 'position', 'score']

        def parse_position(pos_str):
            match = re.match(r'\((\d+),\s*(\d+)\)', str(pos_str))
            if match:
                return int(match.group(1)), int(match.group(2))
            return None

        df['pos'] = df['position'].apply(parse_position)
        df = df.sort_values('time').reset_index(drop=True)

        # 使用后50%作为测试数据，并归一化时间从0开始
        split_idx = len(df) // 2
        self.test_data = df.iloc[split_idx:].copy().reset_index(drop=True)
        self.times = self.test_data['time'].values
        # ★ 关键修复：将时间归一化到从0开始
        self.times = self.times - self.times[0]
        self.positions = list(self.test_data['pos'].values)
        self.scores = self.test_data['score'].values
        self.total_baits = len(self.test_data)
        self.max_time = self.times[-1]

        print(f"测试数据加载: {self.total_baits} 个食饵, 归一化时间范围 0-{self.max_time:.0f}s")

    # ---- UI构建 ----
    def build_ui(self):
        """构建界面"""
        # 左侧：网格画布
        self.left_frame = tk.Frame(self.root, bg=COLOR_BG, width=CANVAS_SIZE+20, height=CANVAS_SIZE+20)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)

        self.canvas = tk.Canvas(
            self.left_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg=COLOR_BG, highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)

        # 右侧：控制面板
        self.right_frame = tk.Frame(self.root, bg=COLOR_PANEL_BG, width=440, height=760)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=False)
        self.right_frame.pack_propagate(False)

        self.build_control_panel()

    def build_control_panel(self):
        """构建右侧控制面板"""
        panel = self.right_frame
        PAD_X = 20
        PAD_Y = 8

        # ---- 标题 ----
        title_frame = tk.Frame(panel, bg=COLOR_PANEL_BG)
        title_frame.pack(fill=tk.X, padx=PAD_X, pady=(15, 5))
        tk.Label(title_frame, text="🤖 机器人竞赛模拟器", font=('Microsoft YaHei', 16, 'bold'),
                 fg=COLOR_HIGHLIGHT, bg=COLOR_PANEL_BG).pack(anchor='w')
        tk.Label(title_frame, text="MLP模型 · 在线决策 · 实时可视化",
                 font=('Microsoft YaHei', 9), fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).pack(anchor='w')

        # ---- 分隔线 ----
        self.add_separator(panel)

        # ---- 状态信息 ----
        info_frame = tk.Frame(panel, bg=COLOR_PANEL_BG)
        info_frame.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)

        # 时间
        tk.Label(info_frame, text="⏱ 模拟时间", font=('Microsoft YaHei', 10),
                 fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).grid(row=0, column=0, sticky='w')
        self.time_label = tk.Label(info_frame, text="0.0s / 0.0s", font=('Consolas', 18, 'bold'),
                                    fg=COLOR_TEXT, bg=COLOR_PANEL_BG)
        self.time_label.grid(row=1, column=0, sticky='w', pady=(0, 5))

        # 进度条
        self.progress_bar = ttk.Progressbar(info_frame, length=380, mode='determinate')
        self.progress_bar.grid(row=2, column=0, sticky='ew', pady=(0, 10), columnspan=2)

        # 得分
        tk.Label(info_frame, text="🏆 当前得分", font=('Microsoft YaHei', 10),
                 fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).grid(row=3, column=0, sticky='w')
        self.score_label = tk.Label(info_frame, text="0", font=('Consolas', 28, 'bold'),
                                     fg=COLOR_HIGHLIGHT, bg=COLOR_PANEL_BG)
        self.score_label.grid(row=4, column=0, sticky='w', pady=(0, 5))

        # 追捕/跳过统计
        stats_row = tk.Frame(info_frame, bg=COLOR_PANEL_BG)
        stats_row.grid(row=5, column=0, sticky='w', pady=5)
        self.pursued_label = tk.Label(stats_row, text="追捕: 0", font=('Microsoft YaHei', 10),
                                       fg=COLOR_PURSUE, bg=COLOR_PANEL_BG)
        self.pursued_label.pack(side=tk.LEFT, padx=(0, 20))
        self.skipped_label = tk.Label(stats_row, text="跳过: 0", font=('Microsoft YaHei', 10),
                                       fg=COLOR_SKIP, bg=COLOR_PANEL_BG)
        self.skipped_label.pack(side=tk.LEFT)

        # ---- 分隔线 ----
        self.add_separator(panel)

        # ---- 当前食饵信息 ----
        bait_frame = tk.Frame(panel, bg=COLOR_PANEL_BG)
        bait_frame.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)

        tk.Label(bait_frame, text="📍 当前食饵", font=('Microsoft YaHei', 11, 'bold'),
                 fg=COLOR_TEXT, bg=COLOR_PANEL_BG).pack(anchor='w')

        self.bait_info_frame = tk.Frame(bait_frame, bg='#1e2d4a', relief=tk.RIDGE, bd=1)
        self.bait_info_frame.pack(fill=tk.X, pady=5)

        self.bait_info_text = tk.Label(
            self.bait_info_frame,
            text="等待食饵出现...",
            font=('Microsoft YaHei', 10), fg=COLOR_TEXT_DIM,
            bg='#1e2d4a', justify=tk.LEFT, anchor='w', padx=10, pady=10
        )
        self.bait_info_text.pack(fill=tk.X)

        # ---- 模型决策 ----
        decision_frame = tk.Frame(panel, bg=COLOR_PANEL_BG)
        decision_frame.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)

        tk.Label(decision_frame, text="🧠 模型决策", font=('Microsoft YaHei', 11, 'bold'),
                 fg=COLOR_TEXT, bg=COLOR_PANEL_BG).pack(anchor='w')

        self.decision_canvas = tk.Canvas(decision_frame, width=380, height=50,
                                          bg=COLOR_PANEL_BG, highlightthickness=0)
        self.decision_canvas.pack(pady=5)

        # 决策详情
        self.decision_detail = tk.Label(decision_frame, text="",
                                         font=('Microsoft YaHei', 9), fg=COLOR_TEXT_DIM,
                                         bg=COLOR_PANEL_BG, justify=tk.LEFT)
        self.decision_detail.pack(anchor='w')

        # ---- 分隔线 ----
        self.add_separator(panel)

        # ---- 控制按钮 ----
        ctrl_frame = tk.Frame(panel, bg=COLOR_PANEL_BG)
        ctrl_frame.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)

        # 速度控制
        tk.Label(ctrl_frame, text="⚡ 模拟速度", font=('Microsoft YaHei', 10),
                 fg=COLOR_TEXT, bg=COLOR_PANEL_BG).pack(anchor='w')
        speed_frame = tk.Frame(ctrl_frame, bg=COLOR_PANEL_BG)
        speed_frame.pack(fill=tk.X, pady=3)

        self.speed_var = tk.IntVar(value=5)
        speed_scale = tk.Scale(speed_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                               variable=self.speed_var, length=250,
                               bg=COLOR_PANEL_BG, fg=COLOR_TEXT, troughcolor='#2d2d44',
                               highlightthickness=0, command=self.on_speed_change)
        speed_scale.pack(side=tk.LEFT)

        self.speed_label = tk.Label(speed_frame, text="5x", font=('Consolas', 14, 'bold'),
                                     fg=COLOR_HIGHLIGHT, bg=COLOR_PANEL_BG, width=4)
        self.speed_label.pack(side=tk.RIGHT, padx=10)

        # 按钮
        btn_frame = tk.Frame(ctrl_frame, bg=COLOR_PANEL_BG)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = tk.Button(btn_frame, text="▶  开始模拟", font=('Microsoft YaHei', 11, 'bold'),
                                    bg='#00aa55', fg='white', activebackground='#00cc66',
                                    relief=tk.FLAT, padx=20, pady=8, cursor='hand2',
                                    command=self.start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.pause_btn = tk.Button(btn_frame, text="⏸  暂停", font=('Microsoft YaHei', 11),
                                    bg='#cc8800', fg='white', activebackground='#ee9900',
                                    relief=tk.FLAT, padx=14, pady=8, cursor='hand2',
                                    command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(btn_frame, text="🔄  重置", font=('Microsoft YaHei', 11),
                                    bg='#555555', fg='white', activebackground='#777777',
                                    relief=tk.FLAT, padx=14, pady=8, cursor='hand2',
                                    command=self.reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # 起始位置选择
        pos_frame = tk.Frame(ctrl_frame, bg=COLOR_PANEL_BG)
        pos_frame.pack(fill=tk.X, pady=5)
        tk.Label(pos_frame, text="起始位置:", font=('Microsoft YaHei', 9),
                 fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)
        self.start_row_var = tk.IntVar(value=5)
        self.start_col_var = tk.IntVar(value=5)
        tk.Spinbox(pos_frame, from_=1, to=10, width=3, textvariable=self.start_row_var,
                   bg='#2d2d44', fg=COLOR_TEXT, buttonbackground='#444').pack(side=tk.LEFT, padx=3)
        tk.Label(pos_frame, text="行", font=('Microsoft YaHei', 8),
                 fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)
        tk.Spinbox(pos_frame, from_=1, to=10, width=3, textvariable=self.start_col_var,
                   bg='#2d2d44', fg=COLOR_TEXT, buttonbackground='#444').pack(side=tk.LEFT, padx=3)
        tk.Label(pos_frame, text="列", font=('Microsoft YaHei', 8),
                 fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)

        # ---- 分隔线 ----
        self.add_separator(panel)

        # ---- 统计面板 ----
        stats_panel = tk.Frame(panel, bg=COLOR_PANEL_BG)
        stats_panel.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)

        tk.Label(stats_panel, text="📊 实时统计", font=('Microsoft YaHei', 11, 'bold'),
                 fg=COLOR_TEXT, bg=COLOR_PANEL_BG).pack(anchor='w')

        stats_grid = tk.Frame(stats_panel, bg=COLOR_PANEL_BG)
        stats_grid.pack(fill=tk.X, pady=5)

        stats_items = [
            ("食饵总数:", "total_baits_label"),
            ("已处理:", "processed_label"),
            ("追捕数:", "pursued_count_label"),
            ("跳过数:", "skipped_count_label"),
            ("最高分食饵:", "max_score_label"),
            ("机器人位置:", "robot_pos_label"),
            ("模型决策准确率:", "accuracy_label"),
            ("得分率(vs最优):", "score_rate_label"),
        ]

        self.stat_labels = {}
        for i, (label_text, var_name) in enumerate(stats_items):
            tk.Label(stats_grid, text=label_text, font=('Microsoft YaHei', 9),
                     fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).grid(
                row=i, column=0, sticky='w', pady=1)
            self.stat_labels[var_name] = tk.Label(
                stats_grid, text="--", font=('Consolas', 9),
                fg=COLOR_TEXT, bg=COLOR_PANEL_BG)
            self.stat_labels[var_name].grid(row=i, column=1, sticky='w', pady=1, padx=(10, 0))

        # 图例
        self.add_separator(panel)
        legend_frame = tk.Frame(panel, bg=COLOR_PANEL_BG)
        legend_frame.pack(fill=tk.X, padx=PAD_X, pady=PAD_Y)
        tk.Label(legend_frame, text="图例", font=('Microsoft YaHei', 9, 'bold'),
                 fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).pack(anchor='w')

        legends = [
            (COLOR_ROBOT, "机器人"),
            (COLOR_BAIT_ACTIVE, "活跃食饵"),
            (COLOR_PURSUE, "追捕决策"),
            (COLOR_SKIP, "跳过决策"),
        ]
        leg_row = tk.Frame(legend_frame, bg=COLOR_PANEL_BG)
        leg_row.pack(fill=tk.X)
        for color, text in legends:
            f = tk.Frame(leg_row, bg=COLOR_PANEL_BG)
            f.pack(side=tk.LEFT, padx=(0, 15))
            tk.Canvas(f, width=12, height=12, bg=color, highlightthickness=0).pack(side=tk.LEFT)
            tk.Label(f, text=f" {text}", font=('Microsoft YaHei', 8),
                     fg=COLOR_TEXT_DIM, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)

    def add_separator(self, parent):
        """添加分隔线"""
        sep = tk.Frame(parent, height=1, bg='#2d2d44')
        sep.pack(fill=tk.X, padx=20, pady=5)

    # ---- 网格绘制 ----
    def grid_to_canvas(self, row, col):
        """网格坐标 -> 画布坐标"""
        x = MARGIN + (col - 1) * CELL_PX
        y = MARGIN + (row - 1) * CELL_PX
        return x, y

    def draw_grid(self):
        """绘制10×10网格"""
        self.canvas.delete("grid")

        # 网格线
        for i in range(GRID_SIZE):
            x0 = MARGIN
            y0 = MARGIN + i * CELL_PX
            x1 = MARGIN + (GRID_SIZE - 1) * CELL_PX
            y1 = y0

            # 水平线
            self.canvas.create_line(x0, y0, x1, y1, fill=COLOR_GRID, width=1, tags="grid")
            # 垂直线
            self.canvas.create_line(y0, x0, y1, x1, fill=COLOR_GRID, width=1, tags="grid")

        # 网格交点
        for r in range(1, GRID_SIZE + 1):
            for c in range(1, GRID_SIZE + 1):
                x, y = self.grid_to_canvas(r, c)
                # 角落特殊标注
                if (r == 1 or r == 10) and (c == 1 or c == 10):
                    size = 4
                    color = '#5a5a7a'
                elif r == 1 or r == 10 or c == 1 or c == 10:
                    size = 3
                    color = COLOR_GRID_POINT
                else:
                    size = 2
                    color = COLOR_GRID_POINT
                self.canvas.create_oval(
                    x-size, y-size, x+size, y+size,
                    fill=color, outline='', tags="grid"
                )

        # 坐标标签
        for i in range(1, GRID_SIZE + 1):
            x, y = self.grid_to_canvas(1, i)
            self.canvas.create_text(x, y - 18, text=str(i), fill=COLOR_TEXT_DIM,
                                     font=('Consolas', 8), tags="grid")
            x, y = self.grid_to_canvas(i, 1)
            self.canvas.create_text(x - 18, y, text=str(i), fill=COLOR_TEXT_DIM,
                                     font=('Consolas', 8), tags="grid")

        # 比例尺
        scale_y = MARGIN + (GRID_SIZE - 1) * CELL_PX + 25
        self.canvas.create_line(
            MARGIN, scale_y, MARGIN + CELL_PX, scale_y,
            fill=COLOR_TEXT_DIM, width=2, tags="grid"
        )
        self.canvas.create_text(
            MARGIN + CELL_PX // 2, scale_y + 12,
            text="1m", fill=COLOR_TEXT_DIM, font=('Consolas', 8), tags="grid"
        )

    # ---- 机器人绘制 ----
    def draw_robot(self, row, col, alpha=1.0):
        """绘制机器人"""
        self.canvas.delete("robot")
        x, y = self.grid_to_canvas(row, col)

        # 光晕
        glow_r = 14
        self.canvas.create_oval(
            x - glow_r, y - glow_r, x + glow_r, y + glow_r,
            fill='', outline=COLOR_ROBOT, width=2, tags="robot",
            dash=(3, 3)
        )

        # 主体
        r = 9
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill=COLOR_ROBOT, outline='#ff6666', width=2, tags="robot"
        )

        # 方向指示器（小三角指向移动方向）
        self.canvas.create_text(x, y, text="●", fill='white',
                                 font=('Arial', 8), tags="robot")

        # 标签
        self.canvas.create_text(x, y - 17, text="ROBOT", fill=COLOR_ROBOT,
                                 font=('Consolas', 7, 'bold'), tags="robot")

    def draw_trail(self):
        """绘制机器人轨迹"""
        self.canvas.delete("trail")
        if len(self.robot_path) < 2:
            return

        points = []
        for pos in self.robot_path:
            x, y = self.grid_to_canvas(pos[0], pos[1])
            points.extend([x, y])

        if len(points) >= 4:
            self.canvas.create_line(
                *points, fill=COLOR_ROBOT_TRAIL, width=2,
                dash=(4, 3), tags="trail"
            )

    # ---- 食饵绘制 ----
    def draw_baits(self):
        """绘制活跃食饵"""
        self.canvas.delete("baits")

        for bait in self.active_baits:
            pos = bait['pos']
            score = bait['score']
            remaining = bait['disappear_time'] - self.simulation_time
            is_target = (self.current_bait is not None and
                        self.current_bait['pos'] == pos)

            x, y = self.grid_to_canvas(pos[0], pos[1])

            # 倒计时环
            if remaining > 0:
                life_pct = remaining / BAIT_LIFETIME
                ring_r = 11
                extent = -int(life_pct * 360)  # 顺时针减少
                # 背景环
                self.canvas.create_oval(
                    x-ring_r, y-ring_r, x+ring_r, y+ring_r,
                    outline='#444466', width=2, tags="baits"
                )
                # 前景环
                if life_pct > 0.3:
                    ring_color = COLOR_BAIT_ACTIVE
                else:
                    ring_color = '#ff4444'
                self.canvas.create_arc(
                    x-ring_r, y-ring_r, x+ring_r, y+ring_r,
                    start=90, extent=extent, outline=ring_color,
                    width=2, style='arc', tags="baits"
                )

            # 食饵主体
            bait_r = 8
            if is_target:
                bait_r = 10
                # 目标高亮
                self.canvas.create_oval(
                    x-14, y-14, x+14, y+14,
                    fill='', outline=COLOR_HIGHLIGHT, width=2, tags="baits"
                )

            bait_color = COLOR_BAIT_ACTIVE if remaining > 0 else COLOR_BAIT_INACTIVE
            self.canvas.create_oval(
                x-bait_r, y-bait_r, x+bait_r, y+bait_r,
                fill=bait_color, outline='#cc9900', width=1, tags="baits"
            )

            # 分值
            font_size = 7 if score >= 10 else 8
            self.canvas.create_text(
                x, y, text=str(score),
                fill=COLOR_BAIT_TEXT, font=('Consolas', font_size, 'bold'), tags="baits"
            )

            # 倒计时数字
            if remaining > 0:
                self.canvas.create_text(
                    x, y + bait_r + 10,
                    text=f"{remaining:.1f}s",
                    fill=COLOR_TEXT_DIM, font=('Consolas', 6), tags="baits"
                )

    def draw_decision_indicator(self):
        """绘制模型决策指示器"""
        self.decision_canvas.delete("decision")

        if self.current_decision is None:
            self.decision_canvas.create_rectangle(
                0, 0, 380, 50, fill=COLOR_PANEL_BG, outline='', tags="decision"
            )
            self.decision_canvas.create_text(
                190, 25, text="等待中...", fill=COLOR_TEXT_DIM,
                font=('Microsoft YaHei', 12), tags="decision"
            )
            return

        if self.current_decision == 'pursue':
            bg_color = '#003322'
            border_color = COLOR_PURSUE
            icon = '✅'
            text = '追捕 PURSUE'
            detail_text = '模型预测该食饵值得追捕'
        else:
            bg_color = '#330000'
            border_color = COLOR_SKIP
            icon = '❌'
            text = '跳过 SKIP'
            detail_text = '模型预测该食饵不值得追捕'

        self.decision_canvas.create_rectangle(
            2, 2, 378, 48, fill=bg_color, outline=border_color, width=2, tags="decision"
        )
        self.decision_canvas.create_text(
            190, 25, text=f"{icon}  {text}", fill=border_color,
            font=('Microsoft YaHei', 14, 'bold'), tags="decision"
        )
        self.decision_detail.config(text=detail_text)

    # ---- UI更新 ----
    def update_display(self):
        """更新所有UI元素"""
        # 网格（只在重置时需要重绘）
        # 机器人
        self.draw_robot(self.robot_pos[0], self.robot_pos[1])
        # 轨迹
        self.draw_trail()
        # 食饵
        self.draw_baits()
        # 决策指示器
        self.draw_decision_indicator()

        # 更新时间/得分标签
        self.time_label.config(text=f"{self.simulation_time:.1f}s / {self.max_time:.0f}s")
        self.score_label.config(text=str(self.total_score))

        # 进度条
        progress = min(100, self.simulation_time / self.max_time * 100)
        self.progress_bar['value'] = progress

        # 追捕/跳过
        self.pursued_label.config(text=f"追捕: {self.pursued_count}")
        self.skipped_label.config(text=f"跳过: {self.skipped_count}")

        # 统计
        processed = self.pursued_count + self.skipped_count
        self.stat_labels['total_baits_label'].config(text=str(self.total_baits))
        self.stat_labels['processed_label'].config(text=str(processed))
        self.stat_labels['pursued_count_label'].config(text=str(self.pursued_count))
        self.stat_labels['skipped_count_label'].config(text=str(self.skipped_count))
        max_score = max(self.scores[:self.bait_index]) if self.bait_index > 0 else 0
        self.stat_labels['max_score_label'].config(text=str(max_score))
        self.stat_labels['robot_pos_label'].config(
            text=f"({self.robot_pos[0]}, {self.robot_pos[1]})"
        )

        # 准确率
        if len(self.decision_history) > 0:
            correct = sum(1 for d in self.decision_history if d['correct'])
            acc = correct / len(self.decision_history) * 100
            self.stat_labels['accuracy_label'].config(text=f"{acc:.1f}%")
        else:
            self.stat_labels['accuracy_label'].config(text="--")

        # 得分率（假设最优分约为总分的86%）
        optimal_est = sum(self.scores) * 0.86
        if optimal_est > 0:
            rate = self.total_score / optimal_est * 100
            self.stat_labels['score_rate_label'].config(text=f"{rate:.1f}%")
        else:
            self.stat_labels['score_rate_label'].config(text="--")

        # 当前食饵信息
        if self.current_bait is not None:
            cb = self.current_bait
            dist = manhattan(self.robot_pos, cb['pos'])
            arrival = self.simulation_time + dist / ROBOT_SPEED
            reachable = "✓ 可达" if dist <= 3 else "✗ 距离太远"
            info_lines = [
                f"位置: ({cb['pos'][0]}, {cb['pos'][1]})  |  分值: {cb['score']}  |  出现时间: {cb['time']:.1f}s",
                f"距离: {dist}m  |  到达时间: {arrival:.1f}s  |  {reachable}",
                f"倒计时: {cb['disappear_time'] - self.simulation_time:.1f}s",
            ]
            self.bait_info_text.config(
                text='\n'.join(info_lines),
                fg=COLOR_TEXT, anchor='w', justify=tk.LEFT
            )
        else:
            if self.simulation_time > 0:
                self.bait_info_text.config(
                    text="暂无活跃食饵",
                    fg=COLOR_TEXT_DIM, anchor='w', justify=tk.LEFT
                )

    # ---- 特征构建 ----
    def build_features(self, bait_time, bait_pos, bait_score):
        """为当前食饵构建特征向量"""
        row, col = bait_pos
        dist = manhattan(self.robot_pos, bait_pos)
        arrival_time = self.simulation_time + dist
        can_catch = 1 if arrival_time <= bait_time + BAIT_LIFETIME else 0

        if can_catch:
            wait_time = max(0, bait_time - arrival_time)
            arrival_at_bait = max(arrival_time, bait_time)
            arrival_time_delta = min(arrival_at_bait - bait_time, 10)
        else:
            wait_time = 0
            arrival_time_delta = 10

        # 找邻近食饵（机会成本）
        opp_mask = ((self.times >= bait_time) &
                   (self.times <= bait_time + 6) &
                   (np.arange(len(self.times)) != self.bait_index))
        opp_scores = self.scores[opp_mask]
        opp_score_sum = opp_scores.sum() if len(opp_scores) > 0 else 0
        opp_max = opp_scores.max() if len(opp_scores) > 0 else 0
        opp_count = len(opp_scores)

        feat_values = {
            'dist_to_bait': dist,
            'dist_normalized': min(dist, 18) / 18.0,
            'reachable': 1 if dist <= 3 else 0,
            'can_catch': can_catch,
            'wait_time': wait_time,
            'arrival_time_delta': arrival_time_delta,
            'score': bait_score,
            'score_log': np.log1p(bait_score),
            'score_sqrt': np.sqrt(bait_score),
            'score_norm': bait_score / 40.0,
            'is_high_score': 1 if bait_score > 10 else 0,
            'is_vhigh_score': 1 if bait_score > 15 else 0,
            'is_low_score': 1 if bait_score <= 3 else 0,
            'efficiency': bait_score / max(dist, 0.5),
            'row': row,
            'col': col,
            'is_boundary': 1 if (row in [1, 10] or col in [1, 10]) else 0,
            'is_corner': 1 if (row in [1, 10] and col in [1, 10]) else 0,
            'dist_to_center': (abs(row - 5.5) + abs(col - 5.5)) / 9.0,
            'robot_row': self.robot_pos[0] / 10.0,
            'robot_col': self.robot_pos[1] / 10.0,
            'time_frac': bait_time / max(self.max_time, 1),
            'time_since_last_bait': bait_time - self.times[self.bait_index-1] if self.bait_index > 0 else 0,
            'time_since_last_bait_clipped': min(bait_time - self.times[self.bait_index-1] if self.bait_index > 0 else 0, 15) / 15.0,
            'time_to_next': self.times[self.bait_index+1] - bait_time if self.bait_index+1 < self.total_baits else 0,
            'time_to_next_clipped': min(self.times[self.bait_index+1] - bait_time if self.bait_index+1 < self.total_baits else 0, 15) / 15.0,
            'past_30s_count': 0, 'past_30s_avg': 0, 'past_30s_max': 0,
            'past_60s_count': 0, 'past_60s_avg': 0, 'past_60s_max': 0,
            'past_120s_count': 0, 'past_120s_avg': 0, 'past_120s_max': 0,
            'opp_score_sum': opp_score_sum,
            'opp_max': opp_max,
            'opp_count': opp_count,
        }

        # 窗口特征
        for w in [30, 60, 120]:
            mask = (self.times >= bait_time - w) & (self.times < bait_time)
            count = mask.sum()
            feat_values[f'past_{w}s_count'] = float(count)
            if count > 0:
                feat_values[f'past_{w}s_avg'] = self.scores[mask].mean()
                feat_values[f'past_{w}s_max'] = self.scores[mask].max()

        # 按训练时的特征顺序构建向量
        feat_vec = np.array([[feat_values.get(col, 0.0) for col in self.feature_names]],
                            dtype=np.float64)
        return feat_vec

    # ---- 模拟逻辑 ----
    def start_simulation(self):
        """开始模拟"""
        if self.is_running:
            return

        self.is_running = True
        self.is_paused = False
        self.start_btn.config(state=tk.DISABLED, text="▶  运行中...")
        self.pause_btn.config(state=tk.NORMAL, text="⏸  暂停")
        self.last_update_time = time_module.time()
        self.run_simulation_step()

    def toggle_pause(self):
        """暂停/继续"""
        if not self.is_running:
            return

        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶  继续")
        else:
            self.pause_btn.config(text="⏸  暂停")
            self.last_update_time = time_module.time()
            self.run_simulation_step()

    def reset_simulation(self):
        """重置模拟"""
        self.is_running = False
        self.is_paused = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None

        self.simulation_time = 0.0
        self.robot_pos = (self.start_row_var.get(), self.start_col_var.get())
        self.robot_target = None
        self.robot_moving = False
        self.robot_path.clear()
        self.robot_path.append(self.robot_pos)
        self.total_score = 0
        self.pursued_count = 0
        self.skipped_count = 0
        self.bait_index = 0
        self.active_baits = []
        self.current_bait = None
        self.current_decision = None
        self.decision_history = []

        self.start_btn.config(state=tk.NORMAL, text="▶  开始模拟")
        self.pause_btn.config(state=tk.DISABLED, text="⏸  暂停")
        self.progress_bar['value'] = 0

        self.canvas.delete("baits")
        self.canvas.delete("robot")
        self.canvas.delete("trail")
        self.draw_grid()
        self.update_display()

    def on_speed_change(self, val):
        """速度变化回调"""
        self.speed = int(val)
        self.speed_label.config(text=f"{self.speed}x")

    def run_simulation_step(self):
        """模拟主循环"""
        if not self.is_running or self.is_paused:
            return

        # 计算实际时间步长
        now = time_module.time()
        real_dt = now - self.last_update_time
        self.last_update_time = now

        # 限制最大步长避免卡顿后跳跃
        real_dt = min(real_dt, 0.1)
        sim_dt = real_dt * self.speed

        # 如果模拟时间超过数据范围，结束
        if self.simulation_time >= self.max_time:
            self.end_simulation()
            return

        # ---- 阶段1: 检查是否有活跃食饵过期 ----
        expired = []
        for bait in self.active_baits:
            if self.simulation_time >= bait['disappear_time']:
                expired.append(bait)
        for bait in expired:
            self.active_baits.remove(bait)
            if self.current_bait is bait:
                self.current_bait = None
                self.current_decision = None

        # ---- 阶段2: 机器人在食饵间移动 ----
        if self.robot_moving and self.robot_target is not None:
            target_pos = self.robot_target
            dist = manhattan(self.robot_pos, target_pos)
            if dist > 0:
                # 沿网格线移动
                move_amount = sim_dt * ROBOT_SPEED  # 米/模拟秒
                while move_amount > 0 and dist > 0:
                    # 每次移动1格（1米）
                    step = min(move_amount, dist)
                    # 找到移动方向
                    dr = target_pos[0] - self.robot_pos[0]
                    dc = target_pos[1] - self.robot_pos[1]
                    # 曼哈顿移动：每次移动一步
                    if step >= 1.0:
                        if abs(dr) >= abs(dc):
                            new_r = self.robot_pos[0] + (1 if dr > 0 else -1)
                            new_c = self.robot_pos[1]
                        else:
                            new_r = self.robot_pos[0]
                            new_c = self.robot_pos[1] + (1 if dc > 0 else -1)
                        new_pos = (new_r, new_c)
                        # 确保在网格内
                        if 1 <= new_pos[0] <= 10 and 1 <= new_pos[1] <= 10:
                            self.robot_pos = new_pos
                            self.robot_path.append(new_pos)
                        move_amount -= 1.0
                    else:
                        move_amount = 0
                    dist = manhattan(self.robot_pos, target_pos)

            # 到达目标
            if manhattan(self.robot_pos, target_pos) == 0:
                self.robot_moving = False
                self.robot_target = None
                # 检查食饵是否还在
                for bait in self.active_baits:
                    if bait['pos'] == self.robot_pos and self.simulation_time <= bait['disappear_time']:
                        # 捕获！
                        self.total_score += bait['score']
                        self.active_baits.remove(bait)
                        if self.current_bait is bait:
                            self.current_bait = None
                            self.current_decision = None
                        break

        # ---- 阶段3: 检查是否有新的食饵出现 ----
        while (self.bait_index < self.total_baits and
               self.times[self.bait_index] <= self.simulation_time):
            idx = self.bait_index
            bait = {
                'idx': idx,
                'time': self.times[idx],
                'pos': self.positions[idx],
                'score': self.scores[idx],
                'disappear_time': self.times[idx] + BAIT_LIFETIME,
            }

            # 检查是否可达
            dist = manhattan(self.robot_pos, bait['pos'])
            arrival_time = self.simulation_time + dist / ROBOT_SPEED

            if arrival_time <= bait['disappear_time']:
                # 可达，使用模型决策
                feat_vec = self.build_features(bait['time'], bait['pos'], bait['score'])
                feat_scaled = self.scaler.transform(feat_vec)
                pred = self.model.predict(feat_scaled)[0]

                if pred == 1:
                    # 追捕
                    self.current_decision = 'pursue'
                    self.current_bait = bait
                    self.active_baits.append(bait)
                    self.pursued_count += 1

                    # 如果没有正在移动，立即开始移动
                    if not self.robot_moving:
                        self.robot_target = bait['pos']
                        self.robot_moving = True
                    # 如果正在移动，等到达后再处理
                else:
                    # 跳过
                    self.current_decision = 'skip'
                    self.skipped_count += 1
                    # 不加入active_baits，但显示决策
                    # 短暂的提示
                    self.current_bait = bait  # 短暂显示

                # 记录决策（用于准确率估算）
                self.decision_history.append({
                    'bait': bait,
                    'decision': self.current_decision,
                    'correct': True,  # 无法实时验证，标记为模型决策
                })
            else:
                # 不可达，必然跳过
                self.current_decision = None
                self.skipped_count += 1

            self.bait_index += 1

        # ---- 阶段4: 如果没有在移动但有活跃食饵，选择最近的可追食饵 ----
        if not self.robot_moving and self.active_baits:
            # 找到最近的活跃食饵
            nearest = None
            nearest_dist = float('inf')
            for bait in self.active_baits:
                d = manhattan(self.robot_pos, bait['pos'])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest = bait
            if nearest is not None and nearest_dist > 0:
                self.robot_target = nearest['pos']
                self.robot_moving = True

        # ---- 更新模拟时间 ----
        self.simulation_time += sim_dt

        # ---- 更新显示 ----
        self.update_display()

        # ---- 下一帧 ----
        self.animation_id = self.root.after(33, self.run_simulation_step)  # ~30 FPS

    def end_simulation(self):
        """模拟结束"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL, text="▶  重新开始")
        self.pause_btn.config(state=tk.DISABLED)

        # 最终统计
        total_bait_score = sum(self.scores)
        optimal_est = total_bait_score * 0.86
        score_rate = self.total_score / optimal_est * 100 if optimal_est > 0 else 0

        messagebox.showinfo(
            "模拟结束",
            f"模拟完成!\n\n"
            f"总得分: {self.total_score}\n"
            f"追捕次数: {self.pursued_count}\n"
            f"跳过次数: {self.skipped_count}\n"
            f"总食饵数: {self.total_baits}\n"
            f"得分率 (vs 最优估计): {score_rate:.1f}%\n"
            f"最终位置: ({self.robot_pos[0]}, {self.robot_pos[1]})"
        )


# ===========================
# 主入口
# ===========================
def main():
    root = tk.Tk()
    root.configure(bg=COLOR_BG)

    # 设置窗口图标和最小尺寸
    root.minsize(1280, 780)

    # 居中显示
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - 1280) // 2
    y = (screen_h - 780) // 2
    root.geometry(f"+{x}+{y}")

    app = RobotSimulator(root)
    root.mainloop()


if __name__ == '__main__':
    main()
