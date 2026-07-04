# -*- coding: utf-8 -*-
"""
任务2：食饵拦截 Gymnasium 环境 BaitGridEnv
====================================
物理规则（题面）：9m×9m 场地，10×10 网格线（间隔1m），食饵在交点出现、停留3秒消失，
机器人 1m/s 只能沿网格线移动。

- 动作：连续方向向量 (ax, ay)∈[-1,1]²，投影到网格线（交点处选点积最大的边，边上只能进/退）
- 观测：机器人位置 + 最近 K=6 个在场食饵(Δx,Δy,v,τ,可达) + 距上次出现时间，共33维
- 奖励：拾取分值 + 基于势的塑形（0.05×最近可达食饵曼哈顿距离减少量）
- 数据：真实回放（前80%训练/后20%验证）+ 可选任务1生成模型增广

自检: D:/conda/envs/AI/python.exe bait_env.py --selftest
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

INPUT = r'input/202607032000建模复赛题附件.xlsx'
GRID = 9              # 交点坐标 0..9（米）
LIFETIME = 3.0
SPEED = 1.0
DT = 0.25             # 决策步长（秒）
K_OBS = 10
EPS = 1e-9
HOT_A = np.array([1.5, 7.0])    # 热点A质心（env 坐标）
HOT_B = np.array([7.5, 1.5])    # 热点B质心

# 任务1拟合的生成模型参数（合成增广用）
GAP_MU, GAP_SIGMA = 6.045, 1.878
VAL_P, VAL_B, VAL_MAX = 0.1089, 1.182, 40
TRAIN_FRAC = 0.8


def load_events(path=INPUT):
    df = pd.read_excel(path, sheet_name='Sheet1')
    df.columns = ['t', 'pos', 'v']
    xy = df['pos'].str.extract(r'\((\d+),(\d+)\)').astype(int)
    # 数据坐标 1..10 → 环境坐标 0..9
    return np.column_stack([df.t.values.astype(float),
                            xy[0].values - 1, xy[1].values - 1,
                            df.v.values.astype(float)])


def value_pmf():
    """奇偶增强截断几何 pmf(k) ∝ p(1-p)^(k-1)·b^(k为奇数)，k=1..40。"""
    k = np.arange(1, VAL_MAX + 1)
    w = VAL_P * (1 - VAL_P) ** (k - 1) * np.where(k % 2 == 1, VAL_B, 1.0)
    return w / w.sum()


class BaitGridEnv(gym.Env):
    """split: 'train'（前80%随机窗口，可混合成序列）| 'val'（后20%滑窗，纯真实数据）"""
    metadata = {'render_modes': []}

    def __init__(self, split='train', episode_len=600.0, synthetic=True,
                 seed=0, val_windows=10):
        super().__init__()
        self.split = split
        self.episode_len = float(episode_len)
        self.synthetic = synthetic and split == 'train'
        self.rng = np.random.default_rng(seed)

        ev = load_events()
        t_cut = ev[-1, 0] * TRAIN_FRAC
        self.train_ev = ev[ev[:, 0] <= t_cut]
        self.val_ev = ev[ev[:, 0] > t_cut]
        # 空间经验分布（仅用训练段估计，避免信息泄漏）
        cells = (self.train_ev[:, 1] * 10 + self.train_ev[:, 2]).astype(int)
        self.cell_p = np.bincount(cells, minlength=100) / len(cells)
        self.val_pmf = value_pmf()

        # 验证滑窗起点（固定，保证可比）
        v0, v1 = self.val_ev[0, 0], self.val_ev[-1, 0]
        n_win = max(1, val_windows)
        span = max(v1 - v0 - self.episode_len, 0)
        self.val_starts = v0 + span * np.linspace(0, 1, n_win)
        self._val_i = 0

        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = spaces.Box(-1, 1, (2 + K_OBS * 5 + 1 + 4,), np.float32)

    # ---------- 序列准备 ----------
    def _sample_synthetic(self):
        n = int(self.episode_len / GAP_MU * 1.5) + 20
        gaps = np.maximum(np.round(self.rng.normal(GAP_MU, GAP_SIGMA, n)), 1)
        t = np.cumsum(gaps)
        cells = self.rng.choice(100, n, p=self.cell_p)
        v = self.rng.choice(np.arange(1, VAL_MAX + 1), n, p=self.val_pmf)
        ev = np.column_stack([t, cells // 10, cells % 10, v.astype(float)])
        return ev[ev[:, 0] <= self.episode_len]

    def _episode_events(self):
        if self.split == 'val':
            t0 = self.val_starts[self._val_i % len(self.val_starts)]
            self._val_i += 1
            ev = self.val_ev
        else:
            if self.synthetic and self.rng.random() < 0.5:
                return self._sample_synthetic()
            ev = self.train_ev
            t0 = ev[0, 0] + self.rng.random() * max(
                ev[-1, 0] - ev[0, 0] - self.episode_len, 0)
        sel = ev[(ev[:, 0] >= t0) & (ev[:, 0] < t0 + self.episode_len)].copy()
        sel[:, 0] -= t0
        return sel

    # ---------- gym 接口 ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.events = self._episode_events()
        self.next_ev = 0
        self.active = []          # [t_appear, x, y, v]
        self.t = 0.0
        self.last_appear = 0.0
        self.pos = np.array([4.0, 4.0])   # 场地中心交点出发
        self.score = 0.0
        self.picked = 0
        self._spawn_and_expire()
        return self._obs(), {}

    def _spawn_and_expire(self):
        while self.next_ev < len(self.events) and self.events[self.next_ev, 0] <= self.t + EPS:
            e = self.events[self.next_ev]
            self.active.append(e.copy())
            self.last_appear = e[0]
            self.next_ev += 1
        self.active = [a for a in self.active if self.t - a[0] < LIFETIME - EPS]

    def _move(self, direction, dist):
        """沿网格线移动 dist 米，返回扫过的线段列表 [(p0, p1), ...]。"""
        segs = []
        p = self.pos.copy()
        remaining = dist
        for _ in range(8):                       # dist=0.5 最多跨1个交点，留裕量
            if remaining <= EPS:
                break
            on_x = abs(p[0] - round(p[0])) < EPS   # x 为整数 → 可沿 y 走
            on_y = abs(p[1] - round(p[1])) < EPS
            if on_x and on_y:                       # 交点：4方向选点积最大
                cands = []
                for d in ([1, 0], [-1, 0], [0, 1], [0, -1]):
                    q = p + d
                    if 0 - EPS <= q[0] <= GRID + EPS and 0 - EPS <= q[1] <= GRID + EPS:
                        cands.append(np.array(d, float))
                dots = [float(direction @ d) for d in cands]
                d = cands[int(np.argmax(dots))]
                if max(dots) <= 0:                  # 所有方向都背离 → 原地不动
                    break
                limit = 1.0                         # 到下一交点
            else:                                   # 边中段：只能沿边进/退
                axis = 1 if on_x else 0             # 自由变化的轴
                s = np.sign(direction[axis])
                if s == 0:
                    break
                d = np.zeros(2); d[axis] = s
                nxt = np.floor(p[axis]) + (1 if s > 0 else 0)
                limit = abs(nxt - p[axis])
                q_end = p[axis] + s * min(limit, remaining)
                if q_end < -EPS or q_end > GRID + EPS:
                    break
            step = min(limit, remaining)
            q = p + d * step
            segs.append((p.copy(), q.copy()))
            p = q
            remaining -= step
        self.pos = np.clip(p, 0, GRID)
        return segs

    def _pickups(self, segs):
        got = 0.0
        for a in list(self.active):
            bx, by = a[1], a[2]
            for p0, p1 in segs:
                # 点在线段上（线段轴对齐）
                if (min(p0[0], p1[0]) - EPS <= bx <= max(p0[0], p1[0]) + EPS and
                        min(p0[1], p1[1]) - EPS <= by <= max(p0[1], p1[1]) + EPS and
                        (abs(p0[0] - p1[0]) < EPS and abs(p0[0] - bx) < EPS or
                         abs(p0[1] - p1[1]) < EPS and abs(p0[1] - by) < EPS)):
                    got += a[3]
                    self.picked += 1
                    self.active = [b for b in self.active if b is not a]
                    break
        return got

    def _potential_target(self):
        """势函数目标：可达食饵中性价比 v/(d+1) 最高者；无可达食饵时为
        A/B 占比加权驻守点（偏向热点A，占比约 0.31:0.13 → 0.7:0.3）。"""
        best, best_cp = None, -1.0
        for a in self.active:
            d = abs(self.pos[0] - a[1]) + abs(self.pos[1] - a[2])
            ttl = LIFETIME - (self.t - a[0])
            if d <= ttl * SPEED + EPS:
                cp = a[3] / (d + 1.0)
                if cp > best_cp:
                    best_cp, best = cp, np.array([a[1], a[2]])
        if best is None:
            best = 0.7 * HOT_A + 0.3 * HOT_B
        return best

    def _phi(self):
        tgt = self._potential_target()
        return abs(self.pos[0] - tgt[0]) + abs(self.pos[1] - tgt[1])

    def step(self, action):
        direction = np.asarray(action, float)
        n = np.linalg.norm(direction)
        phi0 = self._phi()
        segs = self._move(direction / n, SPEED * DT) if n > EPS else []
        reward = self._pickups(segs)
        picked_now = reward > 0
        self.score += reward
        self.t += DT
        self._spawn_and_expire()
        if not picked_now:      # 拾取步不计塑形：势函数目标切换会产生虚假跳变
            reward += 0.1 * (phi0 - self._phi())
        terminated = False
        truncated = self.t >= self.episode_len - EPS
        return self._obs(), float(reward), terminated, truncated, {
            'score': self.score, 'picked': self.picked}

    def _obs(self):
        o = np.zeros(2 + K_OBS * 5 + 1 + 4, np.float32)
        o[0], o[1] = self.pos[0] / GRID, self.pos[1] / GRID
        items = sorted(self.active, key=lambda a: abs(self.pos[0] - a[1]) + abs(self.pos[1] - a[2]))
        for i, a in enumerate(items[:K_OBS]):
            d = abs(self.pos[0] - a[1]) + abs(self.pos[1] - a[2])
            ttl = LIFETIME - (self.t - a[0])
            o[2 + i * 5:2 + i * 5 + 5] = [(a[1] - self.pos[0]) / GRID,
                                          (a[2] - self.pos[1]) / GRID,
                                          a[3] / 40.0, ttl / LIFETIME,
                                          1.0 if d <= ttl * SPEED + EPS else 0.0]
        o[2 + K_OBS * 5] = min((self.t - self.last_appear) / 12.0, 1.0)
        o[-4:] = [(HOT_A[0] - self.pos[0]) / GRID, (HOT_A[1] - self.pos[1]) / GRID,
                  (HOT_B[0] - self.pos[0]) / GRID, (HOT_B[1] - self.pos[1]) / GRID]
        return o


# ============================================================
def _selftest():
    print('== selftest ==')
    # 1. 回放校验：val 环境第一个窗口的事件与原数据一致
    env = BaitGridEnv(split='val', seed=1)
    env.reset()
    raw = env.val_ev
    t0 = env.val_starts[0]
    expect = raw[(raw[:, 0] >= t0) & (raw[:, 0] < t0 + env.episode_len)]
    assert len(env.events) == len(expect) and np.allclose(env.events[:, 1:], expect[:, 1:])
    print(f'回放一致: {len(env.events)} 条事件')

    # 2. 随机策略：始终在线上、无越界、得分一致
    env = BaitGridEnv(split='train', synthetic=False, seed=2)
    obs, _ = env.reset(seed=2)
    total_r, pick_v = 0.0, 0.0
    rng = np.random.default_rng(0)
    done = False
    while not done:
        p = env.pos
        assert -EPS <= p[0] <= GRID + EPS and -EPS <= p[1] <= GRID + EPS
        assert abs(p[0] - round(p[0])) < 1e-6 or abs(p[1] - round(p[1])) < 1e-6, f'离线 {p}'
        s0 = env.score
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
        pick_v += info['score'] - s0
        total_r += r
        done = term or trunc
    assert abs(info['score'] - pick_v) < 1e-6
    print(f'随机策略 600s：得分 {info["score"]:.0f}，拾取 {info["picked"]} 个，'
          f'总奖励 {total_r:.1f}（含塑形）')

    # 3. 拾取判定：食饵在 2m 外剩 2.5s → 直走可拾取
    env = BaitGridEnv(split='train', synthetic=False, seed=3)
    env.reset(seed=3)
    env.active = [np.array([env.t, 6.0, 4.0, 10.0])]
    env.events = env.events[:0]; env.next_ev = 0
    env.pos = np.array([4.0, 4.0])
    got = 0.0
    for _ in range(int(2.0 / (SPEED * DT))):   # 走 2m
        _, r, *_ = env.step(np.array([1.0, 0.0]))
        got += r
    assert got > 9.9, f'应拾取10分, got={got}'
    print('定向拾取判定通过')

    # 4. 合成序列合法性
    env = BaitGridEnv(split='train', synthetic=True, seed=4)
    for _ in range(5):
        env.reset()
        ev = env.events
        assert (ev[:, 3] >= 1).all() and (ev[:, 3] <= 40).all()
        assert (ev[:, 1] >= 0).all() and (ev[:, 1] <= 9).all()
    print('合成序列合法')
    print('== selftest 全部通过 ==')


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        _selftest()
