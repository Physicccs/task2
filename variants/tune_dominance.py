# -*- coding: utf-8 -*-
"""
tune_dominance —— 统治度势场策略（DeepSeek 建议）的调参与配对测试 harness
====================================================================
数据纪律：
  - tune 模式：训练种子 7000–7009（合成 90min × 10 局），多配置选优；
  - test 模式：测试种子 2026 起 --reps 局（默认 30），与 heuristic_v2 /
    your_model 同种子配对对比（报 Δ 均值 ± SE 与配对 t 值），另附
    真实附件 start 894 测试段成绩。
  两组种子不相交 → 调参不污染测试。

用法（从项目根目录运行）：
  D:/conda/envs/AI/python.exe task2/variants/tune_dominance.py --mode tune
  D:/conda/envs/AI/python.exe task2/variants/tune_dominance.py --mode test --reps 30
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK2 = os.path.dirname(_HERE)
for _p in (_HERE, _TASK2, os.path.join(_TASK2, '算法库')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import bait_gen                                     # noqa: E402
import dominance_policy                             # noqa: E402
import hazard_v3                                    # noqa: E402
import pk_task2 as pk                               # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

INPUT = 'input/202607032000建模复赛题附件.xlsx'
TUNE_SEED0, TUNE_REPS, MINUTES = 7000, 10, 90

# 统治度策略候选配置（W_POT, W_HAZ, W_CATCH, W_URG, W_HOME）
DOM_CONFIGS = [
    ('DS原味(Φ_DS+紧迫度)',   dict(W_POT=0,   W_HAZ=1, W_CATCH=0, W_URG=1, W_HOME=0)),
    ('DS原味+可拾引力',        dict(W_POT=0,   W_HAZ=1, W_CATCH=1, W_URG=1, W_HOME=0)),
    ('修正-弱势场',            dict(W_POT=50,  W_HAZ=0, W_CATCH=1, W_URG=0, W_HOME=0)),
    ('修正-强势场',            dict(W_POT=200, W_HAZ=0, W_CATCH=1, W_URG=0, W_HOME=0)),
    ('修正+风险率回撤',        dict(W_POT=50,  W_HAZ=0, W_CATCH=1, W_URG=0, W_HOME=2)),
    ('修正+DS紧迫度',          dict(W_POT=50,  W_HAZ=0, W_CATCH=1, W_URG=1, W_HOME=0)),
    ('修正-拾取主导',          dict(W_POT=10,  W_HAZ=0, W_CATCH=3, W_URG=0, W_HOME=0)),
]
THETA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]            # hazard_v3 的选择性放弃阈值

BEST_DOM = dict(W_POT=0, W_HAZ=1, W_CATCH=1, W_URG=1, W_HOME=0)     # tune 最优：DS原味+可拾引力
BEST_THETA = 0.25    # tune 中 θ>0 全部 ≤0；0.25 是唯一在噪声内的非零值，留作 held-out 复核


def apply_cfg(cls, cfg):
    for k, v in cfg.items():
        setattr(cls, k, v)


def run_seeds(real, cls, seed0, reps):
    out = []
    for r in range(reps):
        ev = bait_gen.generate(MINUTES, seed0 + r, real)
        res = pk.simulate(ev, cls, 0)
        if res['err']:
            raise RuntimeError(f'{cls}: {res["err"]}')
        out.append(res['per_min'])
    return out


def stats(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def paired(xs, base):
    d = [a - b for a, b in zip(xs, base)]
    m, sd = stats(d)
    se = sd / math.sqrt(len(d)) if len(d) > 1 else 0.0
    t = m / se if se > 0 else 0.0
    return m, se, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['tune', 'test'], default='tune')
    ap.add_argument('--reps', type=int, default=30, help='test 模式局数')
    ap.add_argument('--seed0', type=int, default=2026, help='test 模式起始种子')
    args = ap.parse_args()

    from robot_policy import load_events
    real = load_events(INPUT)
    _, v2_cls = pk.load_policy_file('task2/算法库/heuristic_v2.py')

    if args.mode == 'tune':
        print(f'== 调参（训练种子 {TUNE_SEED0}–{TUNE_SEED0 + TUNE_REPS - 1} × {MINUTES}min，'
              f'配对基线 heuristic_v2）==')
        base = run_seeds(real, v2_cls, TUNE_SEED0, TUNE_REPS)
        bm, bs = stats(base)
        print(f'{"heuristic_v2(基线)":<26}{bm:>8.2f} ±{bs:<6.2f}')
        print('\n-- dominance_policy 配置扫描 --')
        for name, cfg in DOM_CONFIGS:
            apply_cfg(dominance_policy.RobotPolicy, cfg)
            xs = run_seeds(real, dominance_policy.RobotPolicy, TUNE_SEED0, TUNE_REPS)
            m, sd = stats(xs)
            dm, se, t = paired(xs, base)
            print(f'{name:<26}{m:>8.2f} ±{sd:<6.2f} | Δvs v2 {dm:+7.2f} ±{se:.2f} (t={t:+.1f})')
        print('\n-- hazard_v3 THETA 扫描（θ=0 ⇔ v2）--')
        for th in THETA_GRID:
            hazard_v3.RobotPolicy.THETA = th
            xs = run_seeds(real, hazard_v3.RobotPolicy, TUNE_SEED0, TUNE_REPS)
            m, sd = stats(xs)
            dm, se, t = paired(xs, base)
            print(f'θ={th:<24}{m:>8.2f} ±{sd:<6.2f} | Δvs v2 {dm:+7.2f} ±{se:.2f} (t={t:+.1f})')
        return

    # ---------------- test ----------------
    apply_cfg(dominance_policy.RobotPolicy, BEST_DOM)
    hazard_v3.RobotPolicy.THETA = BEST_THETA
    _, ym_cls = pk.load_policy_file('task2/算法库/your_model.py')
    pols = [('heuristic_v2', v2_cls), ('your_model', ym_cls),
            (f'dominance{tuple(BEST_DOM.values())}', dominance_policy.RobotPolicy),
            (f'hazard_v3(θ={BEST_THETA})', hazard_v3.RobotPolicy)]

    print(f'== 测试（种子 {args.seed0}–{args.seed0 + args.reps - 1} × {MINUTES}min，'
          f'同种子配对）==')
    res = {n: run_seeds(real, c, args.seed0, args.reps) for n, c in pols}
    base = res['heuristic_v2']
    print(f'\n{"策略":<28}{"μ":>8}{"σ":>7}{"最差":>8}{"Δvs v2":>9}{"SE":>6}{"t":>7}')
    for n, _ in pols:
        m, sd = stats(res[n])
        dm, se, t = paired(res[n], base)
        print(f'{n:<28}{m:>8.2f}{sd:>7.2f}{min(res[n]):>8.2f}'
              f'{dm:>+9.2f}{se:>6.2f}{t:>+7.1f}')

    print(f'\n== 真实附件测试段（start 894）==')
    events = load_events(INPUT)
    for n, c in pols:
        r = pk.simulate(events, c, 893)
        print(f'{n:<28}{r["per_min"]:>8.2f} 分/分钟（总分 {r["score"]:.0f} / 拾取 {r["picked"]}）')


if __name__ == '__main__':
    main()
