# -*- coding: utf-8 -*-
"""
任务2：PPO 训练脚本
====================================
- 8 并行环境（真实回放 + 任务1生成模型增广各半），600s/episode
- EvalCallback 在验证段（真实数据后20%，10个固定滑窗）选 best_model → 早停防过拟合
- 输出: output/task2_ppo/{best_model.zip, final_model.zip, 训练曲线.png, 评估结果.md}

用法:
  D:/conda/envs/AI/python.exe train_task2_ppo.py                  # 全量 3e6 步
  D:/conda/envs/AI/python.exe train_task2_ppo.py --steps 50000    # 冒烟
"""
import argparse
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from bait_env import BaitGridEnv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = 'output/task2_ppo'
TB = 'file/tb_task2'
SEED = 42
EP_LEN = 600.0
N_ENVS = 8


def make_env(split, rank, synthetic=True):
    def _f():
        env = BaitGridEnv(split=split, episode_len=EP_LEN,
                          synthetic=synthetic, seed=SEED + rank)
        return Monitor(env, info_keywords=('score', 'picked'))
    return _f


def evaluate(model, n_ep=10, split='val', deterministic=True, obs_rms=None):
    env = BaitGridEnv(split=split, episode_len=EP_LEN, synthetic=False,
                      seed=SEED + 1000)
    scores, picks = [], []
    for _ in range(n_ep):
        obs, _ = env.reset()
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
        scores.append(info['score']); picks.append(info['picked'])
    return np.array(scores), np.array(picks)


def random_policy_scores(n_ep=10, split='val'):
    env = BaitGridEnv(split=split, episode_len=EP_LEN, synthetic=False, seed=SEED)
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(n_ep):
        _, _ = env.reset()
        done = False
        while not done:
            _, r, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
            done = term or trunc
        scores.append(info['score'])
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=10_000_000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TB, exist_ok=True)

    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[device] {dev}', torch.cuda.get_device_name(0) if dev == 'cuda' else '')

    vec = SubprocVecEnv([make_env('train', i) for i in range(N_ENVS)])
    # 观测已在环境内手工归一化，只做回报归一化（避免 obs_rms 与 best_model 漂移错配）
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=0.9975)
    eval_vec = DummyVecEnv([make_env('val', 100, synthetic=False)])
    # EvalCallback 要求与训练环境同为 VecNormalize；评估壳不做任何归一化
    eval_vec = VecNormalize(eval_vec, training=False, norm_obs=False,
                            norm_reward=False, gamma=0.9975)

    lr_schedule = lambda p: 3e-4 * p          # 线性衰减到 0
    model = PPO('MlpPolicy', vec, seed=SEED, device=dev,
                n_steps=2048, batch_size=512, learning_rate=lr_schedule,
                gamma=0.9975, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=TB, verbose=1)

    cb = EvalCallback(eval_vec, best_model_save_path=OUT,
                      log_path=OUT, eval_freq=max(50_000 // N_ENVS, 1000),
                      n_eval_episodes=10, deterministic=True)
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    model.save(f'{OUT}/final_model')

    # ---------- 评估 ----------
    best = PPO.load(f'{OUT}/best_model', device=dev) \
        if os.path.exists(f'{OUT}/best_model.zip') else model
    val_s, val_p = evaluate(best, 10, 'val')
    train_s, _ = evaluate(best, 10, 'train')
    rand_s = random_policy_scores(10, 'val')

    per_min_val = val_s / (EP_LEN / 60)
    res = dict(
        steps=args.steps,
        val_score_mean=float(val_s.mean()), val_score_std=float(val_s.std()),
        val_per_min=float(per_min_val.mean()),
        val_picked_mean=float(val_p.mean()),
        train_score_mean=float(train_s.mean()),
        random_score_mean=float(rand_s.mean()),
        overfit_gap=float(train_s.mean() - val_s.mean()),
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(f'{OUT}/评估结果.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 训练曲线（EvalCallback 的 evaluations.npz）
    npz = f'{OUT}/evaluations.npz'
    if os.path.exists(npz):
        d = np.load(npz)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        m = d['results'].mean(axis=1)
        sd = d['results'].std(axis=1)
        ax.plot(d['timesteps'], m, 'o-', label='验证段回报（含塑形）')
        ax.fill_between(d['timesteps'], m - sd, m + sd, alpha=0.25)
        ax.axhline(rand_s.mean(), color='gray', ls='--',
                   label=f'随机策略得分 {rand_s.mean():.0f}')
        ax.set_xlabel('训练步数'); ax.set_ylabel('episode 回报')
        ax.set_title('PPO 验证段学习曲线')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{OUT}/任务2_PPO学习曲线.png', dpi=300, bbox_inches='tight')
        print(f'[图] {OUT}/任务2_PPO学习曲线.png')

    # MD 摘要
    with open(f'{OUT}/评估结果.md', 'w', encoding='utf-8') as f:
        f.write(f"""# 任务2 PPO 评估结果

| 指标 | 数值 |
|---|---|
| 训练步数 | {args.steps:,} |
| 验证段（真实数据后20%，10窗）episode 得分 | {val_s.mean():.1f} ± {val_s.std():.1f} |
| **验证段每分钟得分** | **{per_min_val.mean():.2f}** |
| 验证段平均拾取数 / 10分钟 | {val_p.mean():.1f} |
| 训练段得分（同评估协议） | {train_s.mean():.1f} |
| 过拟合差距（训练−验证） | {res['overfit_gap']:.1f} |
| 随机策略验证段得分 | {rand_s.mean():.1f} |
""")
    print('[完成]')


if __name__ == '__main__':
    main()
