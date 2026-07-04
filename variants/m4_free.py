# -*- coding: utf-8 -*-
"""
m4_free —— 机制4（自由探索）：拾取死区修补（脚下扫段 jiggle）+ B区条件驻留 T_stay
====================================================================================
【最优参数与成绩】
  best_params: jiggle=True, t_stay=0, margin=0.0（即只启用死区修补，不启用B区驻留）
  seg A: 36.63 分/分钟（2183分/252拾，基线 33.99 → +2.64）
  seg B: 34.84 分/分钟（1044分/106拾，基线 34.24 → +0.60，只验了一次）
  A+B 合计恢复 157+18 = 175 分 == 诊断死区天花板 175 分（23枚脚下饵全额回收，
  死区事件恰好集中在 seg A）。回归校验：jiggle=False,t_stay=0 复现基线 33.99 逐位一致。

【扫描与证伪记录】
  seg A: jiggle=F/t=0 → 33.99；jiggle=T/t=0 → 36.63；
         jiggle=T, t_stay∈{2,4,6,10} → 全部 36.63（与 t_stay=0 逐位相同）。
  T_stay 无效的机制解释（非"效果小"，而是"触发前提不存在"）：主驻点 S_A=env(2,6)
  下只有 3m 曼哈顿菱形内的食饵会被追击，追击链从不越过 S_A/S_B 中线——对 seg A
  全程 249 次 chase-end 位置逐一检验，0 次严格更近 S_B，min(dB-dA)=+2.0。
  即"在 B 区附近完成拾取"这一驻留前提在单主驻点策略下从不发生（B区饵根本够不到），
  双驻点驻留在此策略结构下是空集；理论上（落点 i.i.d.+平稳）驻留收益
  = T_stay×(rate(S_B)−rate(S_A)) ≤ 0，本就不该开启。

【机制1：拾取死区修补 jiggle（主机制）】
  诊断发现3：全部 893 条事件中有 23 枚 / 175 分错失是"食饵恰好弹在机器人脚下
  （出现瞬间 d==0）"。根因是机制级死区：HeuristicPolicy 对脚下目标返回
  sign(目标-位置)=(0,0) 零动作，而 bait_env._pickups 只对"本步扫过的线段"判定
  拾取——静止不产生线段，脚下食饵在整个 3s 存活期内永远拾不到。
  修补：当排列调度给出的首目标与当前位置重合（d<1e-9）时，朝场地中心方向
  跨一步（sign(4.5-pos)，恒不越界）；0.25m 扫段必然经过该交点 → 当步拾取，
  下一步自然回撤。逐案归因显示这 23 枚几乎无并发、0 例取舍 → 可近乎全额回收，
  天花板 175 分 / 89.6 min ≈ 1.95 分/分钟。
  三个并行同事的方向（等待点/追击门槛+终点感知/最短路内路径塑形）均不含此机制。

【机制2：B区条件驻留 T_stay（分给本机制的独有维度，实证扫描）】
  候选动机：落点分布有强 A 区 env(0..3,6..8) 与弱 B 区 env(7..8,0..3)；机器人在
  B 区附近完成拾取后回主驻点 S_A=env(2,6) 需 ~10s。是否应在 B 区局部覆盖最优点
  S_B（由 seg A 经验分布在 x∈[5,9]×y∈[0,4] 内求 3m 曼哈顿覆盖 argmax）驻留至多
  T_stay 秒再回？
  理论预判（写在实验前）：落点 i.i.d.（前期 26 项检验 0 显著）且过程平稳 →
  "驻留 S_B 再回"与"立即回"的位置轨迹只差一个时间平移，长期收益差
  = T_stay × (rate(S_B) − rate(S_A)) ≤ 0，因为 S_A 是全局覆盖最优点。
  故预计 T_stay=0（即不驻留）最优；扫描 T_stay ∈ {0,2,4,6,10} 予以证实/证伪。

【实现】M4Policy 继承 HeuristicPolicy，追击调度逐行复用 _best_first_target；
  jiggle=False 且 t_stay=0 时 act 与基线逐位等价（用作回归校验）。
  驻留状态机：追击结束（上一步有目标、本步无目标）瞬间，若
  d(pos,S_B)+margin < d(pos,S_A) 则进入 local 模式（deadline=t+T_stay），
  期间空闲目标为 S_B；deadline 到或追击插入后重评。

数据纪律：S_B 仅用 seg_events('A') 估计；seg B 只在选定参数后验证一次。
运行（从项目根目录）：
  D:/conda/envs/AI/python.exe task2/variants/m4_free.py --scan    # seg A 扫参
  D:/conda/envs/AI/python.exe task2/variants/m4_free.py --final --t_stay 0
"""
import sys
import os

sys.path.insert(0, 'task2')
sys.path.insert(0, os.path.join('task2', 'variants'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np  # noqa: E402
from bait_env import GRID, EPS  # noqa: E402
from heuristic_policy import HeuristicPolicy  # noqa: E402

_SB_CACHE = None


def b_station():
    """B 区局部覆盖最优点：仅用 seg A 经验落点分布，在 x∈[5,9]×y∈[0,4] 内
    求 3m 曼哈顿覆盖概率 argmax（并列取期望距离最小）。"""
    global _SB_CACHE
    if _SB_CACHE is None:
        from tune_harness import seg_events
        ev = seg_events('A')
        cells = (ev[:, 1] * 10 + ev[:, 2]).astype(int)
        p = np.bincount(cells, minlength=100) / len(cells)
        cx, cy = np.arange(100) // 10, np.arange(100) % 10
        best, bc, bd = None, -1.0, 1e18
        for wx in range(5, GRID + 1):
            for wy in range(0, 5):
                d = np.abs(cx - wx) + np.abs(cy - wy)
                cov = float(p[d <= 3].sum())
                ed = float((p * d).sum())
                if cov > bc + 1e-12 or (abs(cov - bc) <= 1e-12 and ed < bd):
                    best, bc, bd = np.array([wx, wy], float), cov, ed
        _SB_CACHE = (best, bc)
    return _SB_CACHE


def _md(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


class M4Policy(HeuristicPolicy):
    """机制4策略：jiggle 死区修补 + B区条件驻留。jiggle=False,t_stay=0 ≡ 基线。"""

    def __init__(self, jiggle=True, t_stay=0.0, margin=0.0, train_ev=None):
        super().__init__(train_ev)
        self.jiggle = bool(jiggle)
        self.t_stay = float(t_stay)
        self.margin = float(margin)
        self.sb = b_station()[0] if self.t_stay > 0 else None
        self._mode = 'main'          # 'main' → 空闲回 S_A；'local' → 驻留 S_B
        self._deadline = -1.0
        self._prev_chase = False
        self._inited = False

    def act(self, env):
        tgt = self._best_first_target(env)
        has_chase = tgt is not None

        if not has_chase:
            if self.t_stay > 0:
                # 追击刚结束（或首步）→ 依当前位置重评驻留触发
                if self._prev_chase or not self._inited:
                    if _md(env.pos, self.sb) + self.margin < _md(env.pos, self.wait_point):
                        self._mode = 'local'
                        self._deadline = env.t + self.t_stay
                    else:
                        self._mode = 'main'
                if self._mode == 'local' and env.t >= self._deadline - 1e-9:
                    self._mode = 'main'
                tgt = self.sb if self._mode == 'local' else self.wait_point
            else:
                tgt = self.wait_point
        self._inited = True
        self._prev_chase = has_chase

        delta = tgt - env.pos
        if (self.jiggle and has_chase
                and abs(delta[0]) < 1e-9 and abs(delta[1]) < 1e-9):
            # 死区修补：首目标在脚下 → 朝场地中心跨一步，扫段过点即拾取
            return np.sign(np.array([4.5, 4.5]) - env.pos).astype(np.float32)
        return np.sign(delta).astype(np.float32)


def make_act(jiggle=True, t_stay=0.0, margin=0.0):
    """交付工厂：make_act(**params) → act(env)。"""
    pol = M4Policy(jiggle=jiggle, t_stay=t_stay, margin=margin)
    return pol.act


# ============================================================
def _run(seg, **params):
    from functools import partial
    from tune_harness import evaluate
    r = evaluate(partial(make_act, **params), seg=seg)
    print(f'  seg {seg} | {params} | 得分 {r["score"]:.0f} | 拾取 {r["picked"]} | '
          f'{r["minutes"]:.2f} min | {r["per_min"]:.2f} 分/分钟', flush=True)
    return r


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', action='store_true', help='seg A 扫参')
    ap.add_argument('--final', action='store_true', help='seg B 验证（只跑一次）')
    ap.add_argument('--t_stay', type=float, default=0.0)
    ap.add_argument('--margin', type=float, default=0.0)
    ap.add_argument('--jiggle', type=int, default=1)
    args = ap.parse_args()

    sb, cov = b_station()
    print(f'B区局部驻点 S_B = env({sb[0]:.0f},{sb[1]:.0f})，3m 覆盖 {cov:.3f}'
          f'（仅 seg A 分布估计）')

    if args.scan:
        print('== seg A 扫参 ==')
        _run('A', jiggle=False, t_stay=0.0)          # 回归校验：应 == 基线 33.99
        _run('A', jiggle=True, t_stay=0.0)           # 只开死区修补
        for ts in (2.0, 4.0, 6.0, 10.0):             # 死区修补 + 驻留扫描
            _run('A', jiggle=True, t_stay=ts)
    if args.final:
        print('== seg B 最终验证（只跑一次）==')
        _run('B', jiggle=bool(args.jiggle), t_stay=args.t_stay, margin=args.margin)
