"""Generate the charts used in the task-2/task-3 report:
  1. chart_test_score_rate.png   -- chosen algorithm's cumulative score and
     cumulative score/min on the held-out test set (trained on the first
     893 events only). The robot starts at the trained home cell (free
     preparation time before the match, see report assumptions).
  2. chart_full_top5_control.png -- for the top-5 candidate cells (ranked by
     final, full-dataset control power), how each one's control power
     evolves as cumulative data accumulates over the full 3-hour dataset.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager

for _p in ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    font_manager.fontManager.addfont(_p)

import simulate_strategy as strat

INK = '#1a1a1a'
MUTED = '#6b6b6b'
GRID = '#e3e3e3'
BLUE = '#4C72B0'
PALETTE5 = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

plt.rcParams.update({
    'font.sans-serif': ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.edgecolor': MUTED,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
})

events = strat.load_events('data.xlsx')
train, test = events[:893], events[893:]

# ============================================================
# 1: replay the chosen algorithm on the test set (robot starts at the
#    trained home cell -- free preparation time before the match),
#    logging cumulative score / rate.
# ============================================================
counts = {c: 0 for c in strat.CELLS}
for _, p, v in train:
    counts[p] += 1
total = len(train)
control = strat.control_map(counts, total)
home = max(control, key=control.get)

t0 = test[0][0]
T_end = max(e[0] for e in test) + strat.BAIT_LIFE + 1
by_time = {}
for t, p, v in test:
    by_time.setdefault(t, []).append((p, v))

robot = home
active = []
score = 0.0

mins, cum_score, cum_rate = [], [], []
switch_points = []  # (elapsed_min, new_home) whenever home changes, diagnostic only

for t in range(t0, T_end):
    if t in by_time:
        for pos, val in by_time[t]:
            counts[pos] += 1
            total += 1
            active.append({'pos': pos, 'val': val, 'expire': t + strat.BAIT_LIFE})
        control = strat.control_map(counts, total)
        new_home = max(control, key=control.get)
        if new_home != home:
            switch_points.append(((t - t0) / 60.0, new_home))
        home = new_home

    active = [b for b in active if b['expire'] > t]
    reachable = [b for b in active if strat.manhattan(robot, b['pos']) <= (b['expire'] - t)]
    if reachable:
        reachable.sort(key=lambda b: (-b['val'], strat.manhattan(robot, b['pos']),
                                       -strat.path_score(robot, b['pos'], control)))
        target = reachable[0]['pos']
        robot = strat.step_towards(robot, target)
    else:
        robot = strat.greedy_step(robot, home, control)

    still = []
    for b in active:
        if b['pos'] == robot and b['expire'] > t:
            score += b['val']
        else:
            still.append(b)
    active = still

    elapsed = (t - t0 + 1) / 60.0
    mins.append(elapsed)
    cum_score.append(score)
    cum_rate.append(score / elapsed)

print(f'test run: final score={score:.0f}, final rate={cum_rate[-1]:.2f}/min, '
      f'home switches at: {switch_points}')

# ---- chart 1: cumulative score + cumulative rate ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)

ax1.plot(mins, cum_score, color=BLUE, linewidth=2)
ax1.set_ylabel('累计得分')
ax1.set_title('所选算法在测试集上的收益曲线', fontsize=13, loc='left', pad=12)
ax1.text(mins[-1], cum_score[-1], f'  {cum_score[-1]:.0f}分', color=BLUE,
         va='center', fontsize=10, fontweight='bold')

ax2.plot(mins, cum_rate, color=BLUE, linewidth=2)
ax2.axhline(cum_rate[-1], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
ax2.set_ylabel('累计平均得分/分钟')
ax2.set_xlabel('测试集经过时间（分钟）')
ax2.text(mins[-1], cum_rate[-1], f'  {cum_rate[-1]:.2f}/分钟', color=BLUE,
         va='bottom', fontsize=10, fontweight='bold')

for ax in (ax1, ax2):
    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    ax.margins(x=0.02)

fig.tight_layout()
fig.savefig('chart_test_score_rate.png', dpi=160)
plt.close(fig)

# ============================================================
# 2: top-5 candidate cells' control power over cumulative time
#    on the FULL dataset.
# ============================================================
cnt = {c: 0 for c in strat.CELLS}
for _, p, _ in events:
    cnt[p] += 1
final_control = strat.control_map(cnt, len(events))
top5 = sorted(final_control, key=final_control.get, reverse=True)[:5]
print('top5 candidates (final control power):',
      [(c, round(final_control[c], 4)) for c in top5])

full_counts = {c: 0 for c in strat.CELLS}
full_total = 0
by_time_full = {}
for t, p, v in events:
    by_time_full.setdefault(t, []).append(p)

full_mins = []
series = {c: [] for c in top5}
T_full_end = max(e[0] for e in events) + 1

for t in range(T_full_end):
    if t in by_time_full:
        for p in by_time_full[t]:
            full_counts[p] += 1
            full_total += 1
        full_mins.append(t / 60.0)
        for c in top5:
            val = sum(full_counts[c2] for c2 in strat.NEIGHBORS[c]) / full_total
            series[c].append(val)

fig, ax = plt.subplots(figsize=(8.5, 5.6))
for c, color in zip(top5, PALETTE5):
    ax.plot(full_mins, series[c], color=color, linewidth=1.8, label=f'{c}')

# stagger the end-of-line labels vertically so near-tied final values don't collide
order = sorted(top5, key=lambda c: series[c][-1], reverse=True)
min_gap = 0.014
label_y = {}
prev = None
for c in order:
    y = series[c][-1]
    if prev is not None and prev - y < min_gap:
        y = prev - min_gap
    label_y[c] = y
    prev = y
for c, color in zip(top5, PALETTE5):
    ax.annotate(f'{c}  {series[c][-1]:.3f}', xy=(full_mins[-1], series[c][-1]),
                xytext=(full_mins[-1], label_y[c]), textcoords='data',
                va='center', ha='left', fontsize=9.5, color=color, fontweight='bold',
                annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color=color, lw=0.7, alpha=0.6)
                if abs(label_y[c] - series[c][-1]) > 1e-9 else None)
ax.margins(x=0.02)

ax.set_xlabel('累计时间（分钟，全数据集 0~180 分钟）')
ax.set_ylabel('控制权 control(c)')
ax.set_title('全数据集下 Top-5 候选常驻点的控制权随时间变化', fontsize=13, loc='left', pad=12)
ax.margins(x=0.02)
ax.set_xlim(right=full_mins[-1] * 1.14)
ax.legend(loc='lower right', frameon=False, fontsize=9.5, title='候选格子')

fig.tight_layout()
fig.savefig('chart_full_top5_control.png', dpi=160)
plt.close(fig)

print('charts saved: chart_test_score_rate.png, chart_full_top5_control.png')
