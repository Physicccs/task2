# -*- coding: utf-8 -*-
"""
任务2 合成食饵序列生成器
====================================
按任务1统计分析选定的生成模型采样"非数据集"全新序列
（依据 output/数据分析/任务1_食饵规律分析报告.md，三分量相互独立——
26 项独立性/平稳性检验 BH 校正后 0 显著）：

  - 间隔 g ~ 正态(μ=6.045, σ=1.878) 离散化取整，clamp 到观测支撑 [0, 12]
  - 位置 (x,y) ~ 100 格经验多项分布 π̂（由真实附件数据在线统计，表格坐标 1..10）
  - 分值 v ~ 奇偶增强截断几何：pmf(k) ∝ p(1−p)^(k−1)·b^{1[k为奇数]}，
      p=0.1089, b=1.182, 1 ≤ v ≤ 40（均值 ≈ 8.74）

用途：给赛马场 pk_task2.py 提供合成测试集，检验模型稳健性 / 防止在
题目附件上过拟合——模型可放开在全部真实数据上调参，测试用新生成序列。

零第三方依赖（纯标准库 random）。自检（从项目根目录运行）：
    D:/conda/envs/AI/python.exe task2/bait_gen.py --selftest
"""
import random

GAP_MU, GAP_SIGMA = 6.045, 1.878      # 间隔 ~ 离散化正态
GAP_MIN, GAP_MAX = 0, 12              # 观测支撑（真实数据含 4 处同秒到达）
VAL_P, VAL_ODD_B, VAL_MAX = 0.1089, 1.182, 40   # 分值 ~ 奇偶增强截断几何
GRID = 10


def value_pmf():
    """奇偶增强截断几何 pmf，k=1..40，归一化。"""
    w = [VAL_P * (1 - VAL_P) ** (k - 1) * (VAL_ODD_B if k % 2 == 1 else 1.0)
         for k in range(1, VAL_MAX + 1)]
    s = sum(w)
    return [x / s for x in w]


def pos_weights(real_events):
    """真实事件 → 100 格经验频率 π̂。返回 (格子列表, 权重列表)。"""
    cnt = {}
    for _, x, y, _ in real_events:
        cnt[(int(x), int(y))] = cnt.get((int(x), int(y)), 0) + 1
    cells = [(x, y) for x in range(1, GRID + 1) for y in range(1, GRID + 1)]
    return cells, [cnt.get(c, 0) for c in cells]


def generate(minutes, seed, real_events, first_t=3):
    """生成约 minutes 分钟的合成序列 [(t, x, y, v), ...]，t 为整秒、从 first_t 起。

    real_events 仅用于统计位置经验分布 π̂（间隔/分值用拟合参数采样）。
    同一 (minutes, seed, 数据) 完全可复现。
    """
    rng = random.Random(seed)
    cells, wpos = pos_weights(real_events)
    vals = list(range(1, VAL_MAX + 1))
    pv = value_pmf()
    end = first_t + int(round(minutes * 60))
    events, t = [], first_t
    while t <= end:
        x, y = rng.choices(cells, weights=wpos)[0]
        v = rng.choices(vals, weights=pv)[0]
        events.append((t, x, y, v))
        g = round(rng.gauss(GAP_MU, GAP_SIGMA))
        t += max(GAP_MIN, min(GAP_MAX, g))
    return events


# ---------------- 自检 ----------------
def _selftest():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from robot_policy import load_events
    real = load_events('input/202607032000建模复赛题附件.xlsx')

    ev = generate(600, 42, real)          # 600 分钟长序列，统计量应贴近拟合参数
    gaps = [b[0] - a[0] for a, b in zip(ev, ev[1:])]
    mg = sum(gaps) / len(gaps)
    mv = sum(e[3] for e in ev) / len(ev)
    odd = sum(1 for e in ev if e[3] % 2 == 1) / len(ev)

    assert all(1 <= e[1] <= GRID and 1 <= e[2] <= GRID for e in ev), '坐标越界'
    assert all(isinstance(e[0], int) for e in ev), 't 非整秒'
    assert all(1 <= e[3] <= VAL_MAX for e in ev), '分值越界'
    assert all(g >= 0 for g in gaps), '时间倒流'
    assert 5.7 <= mg <= 6.4, f'间隔均值异常 {mg:.3f}（期望≈6.045）'
    assert 8.2 <= mv <= 9.3, f'分值均值异常 {mv:.3f}（期望≈8.74）'
    assert 0.53 <= odd <= 0.61, f'奇数占比异常 {odd:.3f}（期望≈0.57）'
    assert generate(90, 7, real) == generate(90, 7, real), '同种子不可复现'
    assert generate(90, 7, real) != generate(90, 8, real), '不同种子序列相同'

    # 位置分布贴近经验分布：合成频率与 π̂ 的 L1 距离应远小于均匀分布的
    cells, w = pos_weights(real)
    tot = sum(w)
    cnt = {}
    for _, x, y, _ in ev:
        cnt[(x, y)] = cnt.get((x, y), 0) + 1
    l1_emp = sum(abs(cnt.get(c, 0) / len(ev) - wi / tot) for c, wi in zip(cells, w))
    l1_uni = sum(abs(1 / len(cells) - wi / tot) for wi in w)
    assert l1_emp < 0.5 * l1_uni, f'位置分布偏离经验分布 L1={l1_emp:.3f}（均匀参照 {l1_uni:.3f}）'

    print(f'selftest OK：{len(ev)} 枚 | 间隔均值 {mg:.3f}(≈6.045) | '
          f'分值均值 {mv:.2f}(≈8.74) | 奇数占比 {odd:.3f}(≈0.57) | '
          f'位置 L1 {l1_emp:.3f} < 均匀参照 {l1_uni:.3f} 的一半')


if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
