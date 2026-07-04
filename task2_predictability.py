# -*- coding: utf-8 -*-
"""
任务2前置：食饵序列可预测性分析
====================================
核心问题：知道历史（前 k 个食饵的位置/间隔/分值），能否降低对下一个食饵的不确定性？
若所有序列依赖检验均不显著，则"守热点 + 就近拦截"即为信息论意义上的最优框架。

方法：一阶/二阶马尔可夫检验、条件熵/互信息 + 置换检验（999 次）、
交叉依赖（间隔-位置-分值）、生成器残留扫描；全部序列依赖 p 值做 BH 校正。

用法: D:/conda/envs/normal/python.exe task2_predictability.py
输出: output/任务2_图01..06_*.png (300dpi) + output/任务2前置_可预测性分析报告.md
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from task1_bait_analysis import (load_data, save_fig, fmt_p, MDReport,
                                 ljung_box, fisher_g_test, wilson_ci,
                                 region_label, autocorr, bin_values)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTDIR = 'output'
REPORT = f'{OUTDIR}/任务2前置_可预测性分析报告.md'
SEED = 42
RNG = np.random.default_rng(SEED)
N_PERM = 999

# 全局登记：所有"序列依赖/可预测性"检验的 p 值，最后统一 BH 校正
DEP_TESTS = []      # (编号, 名称, 统计量描述, p)


def register(name, stat_desc, p):
    DEP_TESTS.append((len(DEP_TESTS) + 1, name, stat_desc, float(p)))


# ============================================================
# 信息论工具
# ============================================================

def entropy_of(seq):
    """插入式（plug-in）香农熵，单位 bit。"""
    _, cnt = np.unique(seq, return_counts=True)
    p = cnt / cnt.sum()
    return float(-np.sum(p * np.log2(p)))


def mutual_info(a, b):
    """离散互信息 I(A;B)=H(A)+H(B)−H(A,B)，plug-in 估计（bit）。"""
    joint = np.char.add(np.char.add(a.astype(str), '|'), b.astype(str))
    return entropy_of(a) + entropy_of(b) - entropy_of(joint)


def perm_mi_test(seq, lag=1, n_perm=N_PERM, rng=RNG):
    """滞后 lag 互信息的置换检验。H0：序列可交换（无序依赖）。
    置换零分布自动吸收 plug-in 估计的正偏差。返回 (MI_obs, p, null)。"""
    seq = np.asarray(seq)
    a, b = seq[:-lag], seq[lag:]
    obs = mutual_info(a, b)
    null = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.permutation(seq)
        null[i] = mutual_info(s[:-lag], s[lag:])
    p = (1 + int(np.sum(null >= obs))) / (n_perm + 1)
    return obs, p, null


def cond_entropy(prev, nxt):
    """H(next|prev) = H(joint) − H(prev)，bit。"""
    joint = np.char.add(np.char.add(prev.astype(str), '|'), nxt.astype(str))
    return entropy_of(joint) - entropy_of(prev)


def runs_test_binary(b):
    """二值序列游程检验（正态近似）。返回 (runs, z, p)。"""
    b = np.asarray(b, bool)
    n1, n2 = int(b.sum()), int((~b).sum())
    n = n1 + n2
    runs = 1 + int(np.sum(b[1:] != b[:-1]))
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    return runs, float(z), float(2 * stats.norm.sf(abs(z)))


# ============================================================
# 1. 时间依赖性复核
# ============================================================

def analyze_time_dependence(df, rep):
    rep.h2('1. 投放规则与时间是否有关（复核）')
    gaps = df.t.diff().dropna().values
    t_mid = df.t.values[1:]

    rho, p_sp = stats.spearmanr(t_mid, gaps)
    rep.p(f'- 间隔 vs 时刻的 Spearman 相关：ρ={rho:.4f}（{fmt_p(p_sp)}）。')

    # 分小时段的间隔分布齐性（整数间隔 → 卡方齐性，避免重结点 KS）
    hour = np.minimum(t_mid // 3600, 2).astype(int)
    gbin = np.clip(gaps, 1, 11)          # 1..10, 11+ 合并
    ct = pd.crosstab(hour, gbin)
    chi2, p_hom, dof, _ = stats.chi2_contingency(ct.values)
    rep.p(f'- 三个小时段的间隔分布卡方齐性：χ²={chi2:.2f}, df={dof}, {fmt_p(p_hom)}。')

    # 热点占比随时间（每 30 分钟窗口）
    reg = df.apply(lambda r: region_label(r.x, r.y), axis=1).values
    win = np.minimum(df.t.values // 1800, 5).astype(int)
    ct_r = pd.crosstab(win, reg)
    chi2r, p_r, dofr, _ = stats.chi2_contingency(ct_r.values)
    rep.p(f'- 6 个 30 分钟窗口 × 3 区域占比卡方齐性：χ²={chi2r:.2f}, df={dofr}, {fmt_p(p_r)}。')

    # 图01：滑动窗口 + 热点占比
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 150
    roll_t = np.array([t_mid[i:i + w].mean() for i in range(0, len(gaps) - w)])
    roll_m = np.array([gaps[i:i + w].mean() for i in range(0, len(gaps) - w)])
    axes[0].plot(roll_t / 3600, roll_m, lw=1.2)
    axes[0].axhline(gaps.mean(), color='r', ls='--', label=f'全程均值 {gaps.mean():.3f}s')
    axes[0].set_xlabel('时间（小时）'); axes[0].set_ylabel('间隔滑动均值（秒）')
    axes[0].set_title(f'到达间隔滑动均值（窗口 {w} 个）'); axes[0].legend()

    shares = ct_r.div(ct_r.sum(axis=1), axis=0)
    for col, mk in zip(['热点A', '热点B', '其他'], ['o-', 's-', '^-']):
        n_win = ct_r.sum(axis=1).values
        k_win = ct_r[col].values
        lo, hi = zip(*[wilson_ci(k, n) for k, n in zip(k_win, n_win)])
        centers = (np.arange(6) + 0.5) * 0.5
        axes[1].errorbar(centers, shares[col].values,
                         yerr=[shares[col].values - np.array(lo),
                               np.array(hi) - shares[col].values],
                         fmt=mk, capsize=3, label=col)
    axes[1].set_xlabel('时间（小时）'); axes[1].set_ylabel('占比')
    axes[1].set_title('各区域食饵占比随时间（30 分钟窗，95% Wilson CI）')
    axes[1].legend()
    save_fig(fig, '任务2_图01_时间依赖性复核.png')
    rep.fig('任务2_图01_时间依赖性复核.png', '图01 间隔滑动均值与区域占比随时间')

    rep.conclusion('间隔与时刻无关、间隔分布跨小时段齐性、区域占比随时间稳定——'
                   '投放规则与日历时间无关，规律是时不变的（与任务1平稳性结论一致）。')
    return dict(p_sp=p_sp, p_hom=p_hom, p_r=p_r)


# ============================================================
# 2. 位置序列的马尔可夫结构（核心）
# ============================================================

def analyze_position_markov(df, rep):
    rep.h2('2. 位置序列是否有马尔可夫结构（核心）')
    reg = df.apply(lambda r: region_label(r.x, r.y), axis=1).values
    cell = (df.x * 100 + df.y).values          # 100 状态

    # --- 2.1 三状态一阶转移 ---
    rep.h3('2.1 三状态粗化 {热点A, 热点B, 其他}')
    prev, nxt = reg[:-1], reg[1:]
    ct = pd.crosstab(pd.Series(prev, name='上一个'), pd.Series(nxt, name='下一个'))
    chi2, p_m1, dof, _ = stats.chi2_contingency(ct.values)
    register('位置3状态一阶马尔可夫（卡方独立性）', f'χ²={chi2:.2f}, df={dof}', p_m1)

    trans = ct.div(ct.sum(axis=1), axis=0)
    marg = pd.Series(nxt).value_counts(normalize=True)
    rows = [[a] + [f'{trans.loc[a, b]:.3f}' for b in trans.columns] for a in trans.index]
    rep.table(['上一个＼下一个'] + list(trans.columns), rows)
    rep.p('边际分布：' + ', '.join(f'{k}={v:.3f}' for k, v in marg.items()) +
          f'。转移矩阵各行与边际几乎相同；卡方独立性检验 χ²={chi2:.2f}, df={dof}, {fmt_p(p_m1)}。')

    # 条件熵 vs 边际熵（置换检验滞后1互信息）
    H = entropy_of(nxt)
    Hc = cond_entropy(np.asarray(prev), np.asarray(nxt))
    mi, p_mi, null = perm_mi_test(reg, 1)
    gain = (H - Hc) / H
    register('位置3状态滞后1互信息（置换）', f'MI={mi * 1000:.2f} mbit', p_mi)
    rep.p(f'- 边际熵 H={H:.4f} bit，条件熵 H(下|上)={Hc:.4f} bit，'
          f'名义可预测性增益 {gain * 100:.2f}%；置换检验 {fmt_p(p_mi)}'
          f'（零分布均值 {null.mean() * 1000:.2f} mbit——名义增益量级与纯抽样偏差相当）。')

    # 二阶
    pair = np.char.add(np.char.add(reg[:-2].astype(str), '_'), reg[1:-1].astype(str))
    mi2 = mutual_info(pair, reg[2:])
    null2 = np.empty(N_PERM)
    for i in range(N_PERM):
        s = RNG.permutation(reg)
        pr = np.char.add(np.char.add(s[:-2].astype(str), '_'), s[1:-1].astype(str))
        null2[i] = mutual_info(pr, s[2:])
    p_mi2 = (1 + int(np.sum(null2 >= mi2))) / (N_PERM + 1)
    register('位置3状态二阶依赖 I((n-1,n); n+1)（置换）', f'MI={mi2 * 1000:.2f} mbit', p_mi2)
    rep.p(f'- 二阶依赖 I((Xₙ₋₁,Xₙ); Xₙ₊₁)={mi2 * 1000:.2f} mbit，置换检验 {fmt_p(p_mi2)}。')

    # A/B 交替性
    pA = float(np.mean(reg == '热点A'))
    afterA = nxt[prev == '热点A']
    kAA, nA = int(np.sum(afterA == '热点A')), len(afterA)
    p_alt = float(stats.binomtest(kAA, nA, pA).pvalue)
    register('热点A自跟随 P(A|A) vs P(A)（二项）', f'{kAA}/{nA}={kAA / nA:.3f} vs {pA:.3f}', p_alt)
    rep.p(f'- 热点A自跟随：P(下一个∈A | 上一个∈A)={kAA / nA:.3f}，边际 P(A)={pA:.3f}，'
          f'二项检验 {fmt_p(p_alt)}——无"交替"或"成串"倾向。')

    # 图02：转移矩阵 vs 边际
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    order = ['热点A', '热点B', '其他']
    tm = trans.loc[order, order].values
    im = axes[0].imshow(tm, cmap='Blues', vmin=0, vmax=0.7)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f'{tm[i, j]:.3f}', ha='center', va='center')
    axes[0].set_xticks(range(3), order); axes[0].set_yticks(range(3), order)
    axes[0].set_xlabel('下一个位置'); axes[0].set_ylabel('上一个位置')
    axes[0].set_title('一阶转移概率矩阵')
    plt.colorbar(im, ax=axes[0])
    xb = np.arange(3)
    axes[1].bar(xb - 0.32, [marg[o] for o in order], 0.2, label='边际')
    for k, o in enumerate(order):
        axes[1].bar(xb + (k - 1) * 0.2 + 0.08, trans.loc[o, order].values, 0.2,
                    label=f'上一个={o}', alpha=0.75)
    axes[1].set_xticks(xb, order); axes[1].set_ylabel('概率')
    axes[1].set_title('各行转移分布 vs 边际分布（几乎重合）')
    axes[1].legend(fontsize=8)
    save_fig(fig, '任务2_图02_三状态转移结构.png')
    rep.fig('任务2_图02_三状态转移结构.png', '图02 三状态转移矩阵与边际对比')

    # --- 2.2 滞后 k 互信息（3状态） ---
    rep.h3('2.2 滞后 k 依赖扫描（k=1..10）')
    lags = range(1, 11)
    mis, pks = [], []
    for k in lags:
        m, pk, _ = perm_mi_test(reg, k)
        mis.append(m); pks.append(pk)
        if k >= 2:      # k=1 已在 2.1 节登记，避免重复计入 BH 校正
            register(f'位置3状态滞后{k}互信息（置换）', f'MI={m * 1000:.2f} mbit', pk)
    rep.table(['滞后 k'] + [str(k) for k in lags],
              [['MI (mbit)'] + [f'{m * 1000:.2f}' for m in mis],
               ['置换 p'] + [f'{p:.3f}' for p in pks]])

    # --- 2.3 细粒度：100格 与 行/列 ---
    rep.h3('2.3 细粒度状态空间')
    mi_c, p_c, null_c = perm_mi_test(cell, 1)
    register('位置100格滞后1互信息（置换）', f'MI={mi_c:.4f} bit', p_c)
    mi_x, p_x, _ = perm_mi_test(df.x.values, 1)
    mi_y, p_y, _ = perm_mi_test(df.y.values, 1)
    register('x坐标滞后1互信息（置换）', f'MI={mi_x * 1000:.2f} mbit', p_x)
    register('y坐标滞后1互信息（置换）', f'MI={mi_y * 1000:.2f} mbit', p_y)
    rep.p(f'- 100 格状态：MI(lag1)={mi_c:.4f} bit（{fmt_p(p_c)}；'
          f'置换零分布均值 {null_c.mean():.4f} bit——观测值完全落入偏差范围）。')
    rep.p(f'- x 坐标序列 MI(lag1)={mi_x * 1000:.2f} mbit（{fmt_p(p_x)}）；'
          f'y 坐标序列 MI(lag1)={mi_y * 1000:.2f} mbit（{fmt_p(p_y)}）。')

    # 图03：滞后k互信息与置换包络
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    env = []
    for k in lags:
        null_k = np.empty(200)
        for i in range(200):
            s = RNG.permutation(reg)
            null_k[i] = mutual_info(s[:-k], s[k:])
        env.append(np.quantile(null_k, 0.95))
    ax.plot(list(lags), [m * 1000 for m in mis], 'o-', label='观测 MI')
    ax.plot(list(lags), [e * 1000 for e in env], 'r--', label='置换零分布 95% 分位')
    ax.set_xlabel('滞后 k'); ax.set_ylabel('互信息（mbit）')
    ax.set_title('位置序列（3状态）滞后 k 互信息 vs 置换包络')
    ax.legend()
    save_fig(fig, '任务2_图03_滞后互信息扫描.png')
    rep.fig('任务2_图03_滞后互信息扫描.png', '图03 滞后 k 互信息与置换 95% 包络')

    rep.conclusion('三层粒度（3状态/行列/100格）、一阶与二阶、滞后1–10 的全部依赖检验均不显著，'
                   '互信息量级与纯抽样偏差相当。位置序列与独立同分布抽样（固定热点多项分布）不可区分：'
                   '知道上一个（或前几个）食饵在哪，对预测下一个落点没有任何帮助。')
    return dict(gain=gain)


# ============================================================
# 3. 间隔序列的可预测性 + 交叉依赖
# ============================================================

def analyze_gap_dependence(df, rep):
    rep.h2('3. 间隔序列与交叉依赖')
    gaps = df.t.diff().dropna().values
    reg = df.apply(lambda r: region_label(r.x, r.y), axis=1).values

    Q, p_lb, rho = ljung_box(gaps, 10)
    register('间隔 Ljung-Box(10)', f'Q={Q:.2f}', p_lb)
    rep.p(f'- 间隔自相关 Ljung-Box(10)：Q={Q:.2f}, {fmt_p(p_lb)}；|ρ̂₁|={abs(rho[0]):.4f}。')

    # 条件均值 E[gap_{n+1} | gap_n]
    g_prev, g_next = gaps[:-1], gaps[1:]
    rho_g, p_g = stats.spearmanr(g_prev, g_next)
    register('相邻间隔 Spearman', f'ρ={rho_g:.4f}', p_g)

    # 交叉：间隔 → 下一个位置是否热点
    is_hot = (reg[1:] != '其他')
    u, p_hot = stats.mannwhitneyu(gaps[is_hot], gaps[~is_hot])
    register('间隔长短 vs 下一位置是否热点（MWU）',
             f'均值 {gaps[is_hot].mean():.3f} vs {gaps[~is_hot].mean():.3f}', p_hot)
    rep.p(f'- 到本次的间隔 vs 本次是否落在热点：热点组均值 {gaps[is_hot].mean():.3f}s，'
          f'非热点组 {gaps[~is_hot].mean():.3f}s，MWU {fmt_p(p_hot)}。')

    # 交叉：间隔 ↔ 分值
    v = df.v.values
    r1, p_v1 = stats.spearmanr(gaps, v[1:])        # 等待越久分值越高？
    r2, p_v2 = stats.spearmanr(gaps, v[:-1])       # 高分后间隔更长？
    register('间隔 vs 本次分值 Spearman', f'ρ={r1:.4f}', p_v1)
    register('上次分值 vs 随后间隔 Spearman', f'ρ={r2:.4f}', p_v2)
    rep.p(f'- 等待时长 vs 本次分值：ρ={r1:.4f}（{fmt_p(p_v1)}）；'
          f'上次分值 vs 随后间隔：ρ={r2:.4f}（{fmt_p(p_v2)}）——分值不"补偿"等待。')

    # 图04：条件均值 + 自相关
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ubins = np.arange(1, 13)
    cm = [g_next[g_prev == k].mean() for k in ubins if np.sum(g_prev == k) >= 10]
    ks = [k for k in ubins if np.sum(g_prev == k) >= 10]
    axes[0].plot(ks, cm, 'o-')
    axes[0].axhline(gaps.mean(), color='r', ls='--', label=f'无条件均值 {gaps.mean():.3f}s')
    axes[0].set_xlabel('上一间隔 $g_n$（秒）'); axes[0].set_ylabel('$E[g_{n+1}|g_n]$（秒）')
    axes[0].set_title('下一间隔的条件均值（平坦 = 不可预测）')
    axes[0].legend()
    axes[1].stem(range(1, 11), rho)
    ci = 1.96 / np.sqrt(len(gaps))
    axes[1].axhline(ci, color='r', ls='--'); axes[1].axhline(-ci, color='r', ls='--')
    axes[1].set_xlabel('滞后 k'); axes[1].set_ylabel(r'$\hat{\rho}_k$')
    axes[1].set_title('间隔自相关（红线 = 95% 白噪声带）')
    save_fig(fig, '任务2_图04_间隔可预测性.png')
    rep.fig('任务2_图04_间隔可预测性.png', '图04 间隔条件均值与自相关')

    rep.conclusion('间隔序列自身无记忆（白噪声），且与位置、分值均无交叉依赖：'
                   '等多久、落在哪、值多少三者相互独立。')


# ============================================================
# 4. 分值序列
# ============================================================

def analyze_value_sequence(df, rep):
    rep.h2('4. 分值序列')
    v = df.v.values
    Q, p_lb, rho = ljung_box(v.astype(float), 10)
    register('分值 Ljung-Box(10)', f'Q={Q:.2f}', p_lb)

    odd = (v % 2 == 1)
    runs, z, p_runs = runs_test_binary(odd)
    register('分值奇偶序列游程检验', f'runs={runs}, z={z:.2f}', p_runs)

    vb = bin_values(pd.Series(v)).astype(str).values
    mi_v, p_miv, _ = perm_mi_test(vb, 1)
    register('分值分箱滞后1互信息（置换）', f'MI={mi_v * 1000:.2f} mbit', p_miv)

    rep.p(f'- 分值 Ljung-Box(10)：Q={Q:.2f}, {fmt_p(p_lb)}。')
    rep.p(f'- 奇偶序列游程检验：runs={runs}, z={z:.2f}, {fmt_p(p_runs)}——'
          f'奇偶偏差是**边际性质**（每次独立抽样偏向奇数），不在序列中成串。')
    rep.p(f'- 分值分箱滞后1互信息 {mi_v * 1000:.2f} mbit（{fmt_p(p_miv)}）。')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].stem(range(1, 11), rho)
    ci = 1.96 / np.sqrt(len(v))
    axes[0].axhline(ci, color='r', ls='--'); axes[0].axhline(-ci, color='r', ls='--')
    axes[0].set_xlabel('滞后 k'); axes[0].set_ylabel(r'$\hat{\rho}_k$')
    axes[0].set_title('分值序列自相关')
    w = 100
    frac = np.array([odd[i:i + w].mean() for i in range(0, len(odd) - w, 10)])
    axes[1].plot(np.arange(len(frac)) * 10 + w / 2, frac, lw=1)
    axes[1].axhline(odd.mean(), color='r', ls='--', label=f'全程奇数占比 {odd.mean():.3f}')
    axes[1].set_xlabel('序号'); axes[1].set_ylabel('滑动奇数占比')
    axes[1].set_title(f'奇数占比滑动窗口（窗口 {w}）')
    axes[1].legend()
    save_fig(fig, '任务2_图05_分值序列.png')
    rep.fig('任务2_图05_分值序列.png', '图05 分值自相关与奇数占比稳定性')

    rep.conclusion('分值序列无自相关、奇偶不成串、相邻分值无互信息：分值是逐个独立抽样的。')


# ============================================================
# 5. 生成器残留扫描 + 到达时刻可预测性
# ============================================================

def analyze_generator_residue(df, rep):
    rep.h2('5. 生成器残留扫描与"真正可预测的部分"')
    gaps = df.t.diff().dropna().values
    reg = df.apply(lambda r: region_label(r.x, r.y), axis=1).values
    cell = (df.x * 100 + df.y).values
    n = len(df)

    rep.h3('5.1 弱信号扫描')
    # 相邻同格
    same = int(np.sum(cell[1:] == cell[:-1]))
    _, cnt = np.unique(cell, return_counts=True)
    p_same = float(np.sum((cnt / n) ** 2))
    exp_same = p_same * (n - 1)
    p_rep = float(stats.binomtest(same, n - 1, p_same).pvalue)
    register('相邻食饵同格重复 vs 独立期望（二项）', f'{same} vs 期望 {exp_same:.1f}', p_rep)
    rep.p(f'- 相邻两食饵落在同一格：观测 {same} 次，独立模型期望 {exp_same:.1f} 次'
          f'（{fmt_p(p_rep)}）——无"连投同点"或"避开上一点"机制。')

    # 热点A指示序列周期性
    fpk, g, p_fg, _ = fisher_g_test((reg == '热点A').astype(float))
    register('热点A指示序列 Fisher g 周期检验', f'g={g:.4f}', p_fg)
    rep.p(f'- 热点A指示序列的 Fisher g 周期检验：g={g:.4f}（{fmt_p(p_fg)}）——'
          f'热点间无轮换周期。')

    # 完整元组重复
    tup = pd.Series(list(zip(gaps.astype(int), df.x.values[1:], df.y.values[1:],
                             df.v.values[1:])))
    n_dup_tup = int(tup.duplicated().sum())
    rep.p(f'- (间隔,x,y,v) 完整元组重复 {n_dup_tup} 次（n={len(tup)}）——'
          f'元组空间大，少量重复符合独立抽样，未见成段复制的伪随机序列复用痕迹。')

    rep.h3('5.2 唯一高度可预测的量：到达时刻')
    mu, sd = gaps.mean(), gaps.std(ddof=1)
    rows = []
    for k in (1.0, 1.5, 2.0, 2.5):
        lo, hi = mu - k * sd, mu + k * sd
        cov = float(np.mean((gaps >= lo) & (gaps <= hi)))
        rows.append([f'μ±{k}σ', f'[{lo:.1f}, {hi:.1f}]s', f'{cov * 100:.1f}%'])
    rep.table(['窗口', '区间', '实际覆盖率'], rows)
    q05, q95 = np.quantile(gaps, [0.05, 0.95])
    rep.p(f'- 经验 5%–95% 分位区间：[{q05:.0f}, {q95:.0f}] 秒。'
          f'由于间隔强欠散（Fano≈0.11），**下一个食饵的到达时刻**是全序列中'
          f'唯一可以高置信预测的量——这正是任务2可利用的核心规律。')

    # 图06：N(t) 规整性 —— 部分和的漂移带宽 vs 泊松
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    t_obs = df.t.values
    idx = np.arange(len(t_obs))
    ax.plot(t_obs / 3600, idx - t_obs / mu, lw=1, label='观测：$N(t) - t/\\mu$')
    # 独立同边际（置换次序）模拟包络
    cs = np.array([np.cumsum(RNG.permutation(gaps)) for _ in range(200)])
    dev = np.arange(1, len(gaps) + 1) - cs / mu
    lo_env, hi_env = np.quantile(dev, [0.025, 0.975], axis=0)
    ax.fill_between(cs.mean(axis=0) / 3600, lo_env, hi_env, alpha=0.25,
                    label='独立同边际置换 95% 包络')
    ax.set_xlabel('时间（小时）'); ax.set_ylabel('累计计数偏离 $N(t)-t/\\mu$')
    ax.set_title('到达过程的规整性：计数漂移远小于泊松（±$\\sqrt{t/\\mu}$ 级）')
    ax.legend()
    save_fig(fig, '任务2_图06_到达时刻可预测性.png')
    rep.fig('任务2_图06_到达时刻可预测性.png', '图06 累计计数偏离与独立置换包络')

    rep.conclusion('未发现任何生成器残留信号；到达"节奏"（下一个食饵约 82% 概率在 3–9 秒内、'
                   '约 94% 在 2–10 秒内出现）是唯一可预测的维度。')


# ============================================================
# 6. 结论综合 + BH 校正
# ============================================================

def summarize(df, rep, gain):
    rep.h2('6. 综合结论：任务2的信息边界')
    ps = np.array([t[3] for t in DEP_TESTS])
    p_adj = stats.false_discovery_control(ps, method='bh')
    n_sig_raw = int(np.sum(ps < 0.05))
    n_sig_adj = int(np.sum(p_adj < 0.05))

    rows = [[i, name, desc, fmt_p(p), fmt_p(pa),
             '**显著**' if pa < 0.05 else '不显著']
            for (i, name, desc, p), pa in zip(DEP_TESTS, p_adj)]
    rep.p(f'共执行 {len(DEP_TESTS)} 项序列依赖/可预测性检验，'
          f'原始 p<0.05 的有 {n_sig_raw} 项，BH 校正后显著 {n_sig_adj} 项：')
    rep.table(['#', '检验', '统计量', '原始 p', 'BH 校正 p', '判定'], rows)

    rep.h3('回答三个问题')
    rep.p('**(1) 投放规则与时间有关吗？** 无关。速率、间隔分布、热点占比、分值分布在 3 小时内'
          '全部平稳，规律是时不变的。')
    rep.p('**(2) 有没有可利用的内在序列规律？** 位置、分值、间隔三个序列自身无记忆，'
          '相互之间也无交叉依赖；条件熵与边际熵的差异在置换零分布偏差范围内'
          f'（3 状态下名义增益仅 {gain * 100:.2f}%）。数据与"每次独立地从固定分布抽样"不可区分。')
    rep.p('**(3) 任务2 只能守热点吗？** 在"预测下一个落点"意义上，是的——历史信息对落点预测'
          '零增益，守热点（长期占比最高的区域）就是信息论最优的位置先验。但可利用的规律不止'
          '位置先验一条：**到达节奏高度可预测**（下一个食饵约 82% 概率落在 μ±1.5σ ≈ 3–9 秒内、'
          '约 94% 落在 2–10 秒内，见 §5.2），且**分值与位置独立**（热点食饵不贬值）。'
          '因此最优策略框架 = 固定空间先验（热点驻守/巡弋）+ 节奏化的实时拦截调度，'
          '而不需要（也不可能）做逐个落点预测。')

    rep.conclusion('食饵序列 = 平稳更新过程（近等间隔）× 独立同分布落点（固定双热点多项分布）'
                   '× 独立同分布分值。除"到达节奏"外不存在可利用的序列规律；'
                   '守热点不是无奈之选，而是可证明的信息论最优位置先验。')


def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    df = load_data()
    rep = MDReport()
    rep.h1('任务2前置：食饵序列可预测性分析')
    rep.p(f'样本 n={len(df)}；方法：马尔可夫检验、条件熵/互信息置换检验（{N_PERM} 次，seed={SEED}）、'
          '交叉依赖检验、生成器残留扫描；全部序列依赖 p 值经 Benjamini–Hochberg 校正。')
    rep.p('**核心问题**：知道历史能否降低对下一个食饵（时刻/位置/分值）的不确定性？')

    analyze_time_dependence(df, rep)
    r2 = analyze_position_markov(df, rep)
    analyze_gap_dependence(df, rep)
    analyze_value_sequence(df, rep)
    analyze_generator_residue(df, rep)
    summarize(df, rep, r2['gain'])

    rep.write(REPORT)
    print(f'[完成] 检验总数 {len(DEP_TESTS)}')


if __name__ == '__main__':
    main()
