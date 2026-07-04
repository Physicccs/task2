#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器人竞赛 - 在线推理脚本 v2
============================
输入：食饵数据表（Excel格式，列: 出现时刻, 位置, 分值）
输出：每次食饵刷新时，机器人前进的方向
      方向编码: 右=(1,0)  左=(-1,0)  上=(0,1)  下=(0,-1)  不动=(0,0)

使用训练好的MLP模型进行实时决策。
"""

import pandas as pd
import numpy as np
import re
import sys
import io
import joblib
import json

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================
# 常量
# ============================
ROBOT_SPEED = 1.0
BAIT_LIFETIME = 3.0


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def parse_position(pos_str):
    m = re.match(r'\((\d+),\s*(\d+)\)', str(pos_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def compute_direction(current_pos, target_pos):
    """
    计算从 current_pos 到 target_pos 的第一个曼哈顿步方向。
    优先水平移动（列方向→左右），再垂直移动（行方向→上下）。

    用户坐标系: (x, y)
      右: col+1 → ( 1,  0)
      左: col-1 → (-1,  0)
      上: row-1 → ( 0,  1)
      下: row+1 → ( 0, -1)
      不动:     → ( 0,  0)
    """
    r1, c1 = current_pos
    r2, c2 = target_pos

    if r1 == r2 and c1 == c2:
        return (0, 0)

    # 优先水平移动
    if c1 < c2:
        return (1, 0)    # 向右
    if c1 > c2:
        return (-1, 0)   # 向左

    # 垂直移动
    if r1 < r2:
        return (0, -1)   # 向下（row增大=向下）
    if r1 > r2:
        return (0, 1)    # 向上（row减小=向上）

    return (0, 0)


# ============================
# 特征构建（与训练时 solve.py 的 simulate_online 保持一致）
# ============================
def build_features(robot_pos, robot_time, bait_time, bait_pos, bait_score,
                   bait_index, all_times, all_scores, total_baits):
    """为当前食饵构建38维特征向量"""
    row, col = bait_pos
    dist = manhattan(robot_pos, bait_pos)
    arrival_time = robot_time + dist / ROBOT_SPEED
    can_catch = 1 if arrival_time <= bait_time + BAIT_LIFETIME else 0

    if can_catch:
        wait_time = max(0, bait_time - arrival_time)
        arrival_at_bait = max(arrival_time, bait_time)
        arrival_time_delta = min(arrival_at_bait - bait_time, 10)
    else:
        wait_time = 0
        arrival_time_delta = 10

    # 机会成本
    opp_mask = ((all_times >= bait_time) &
                (all_times <= bait_time + 6) &
                (np.arange(len(all_times)) != bait_index))
    opp_scores = all_scores[opp_mask]
    opp_score_sum = float(opp_scores.sum()) if len(opp_scores) > 0 else 0.0
    opp_max = float(opp_scores.max()) if len(opp_scores) > 0 else 0.0
    opp_count = len(opp_scores)

    # 使用训练集的 max_time（5370s）来归一化 time_frac，保证与训练分布一致
    TRAIN_MAX_TIME = 5370.0

    feat_values = {
        'dist_to_bait': float(dist),
        'dist_normalized': min(dist, 18) / 18.0,
        'reachable': 1 if dist <= 3 else 0,
        'can_catch': can_catch,
        'wait_time': float(wait_time),
        'arrival_time_delta': float(arrival_time_delta),
        'score': float(bait_score),
        'score_log': float(np.log1p(bait_score)),
        'score_sqrt': float(np.sqrt(bait_score)),
        'score_norm': bait_score / 40.0,
        'is_high_score': 1 if bait_score > 10 else 0,
        'is_vhigh_score': 1 if bait_score > 15 else 0,
        'is_low_score': 1 if bait_score <= 3 else 0,
        'efficiency': bait_score / max(dist, 0.5),
        'row': float(row),
        'col': float(col),
        'is_boundary': 1 if (row in [1, 10] or col in [1, 10]) else 0,
        'is_corner': 1 if (row in [1, 10] and col in [1, 10]) else 0,
        'dist_to_center': (abs(row - 5.5) + abs(col - 5.5)) / 9.0,
        'robot_row': robot_pos[0] / 10.0,
        'robot_col': robot_pos[1] / 10.0,
        'time_frac': bait_time / TRAIN_MAX_TIME,
        'time_since_last_bait': float(bait_time - all_times[bait_index-1]) if bait_index > 0 else 0.0,
        'time_since_last_bait_clipped': min(float(bait_time - all_times[bait_index-1]) if bait_index > 0 else 0.0, 15) / 15.0,
        'time_to_next': float(all_times[bait_index+1] - bait_time) if bait_index + 1 < total_baits else 0.0,
        'time_to_next_clipped': min(float(all_times[bait_index+1] - bait_time) if bait_index + 1 < total_baits else 0.0, 15) / 15.0,
        'past_30s_count': 0.0, 'past_30s_avg': 0.0, 'past_30s_max': 0.0,
        'past_60s_count': 0.0, 'past_60s_avg': 0.0, 'past_60s_max': 0.0,
        'past_120s_count': 0.0, 'past_120s_avg': 0.0, 'past_120s_max': 0.0,
        'opp_score_sum': opp_score_sum,
        'opp_max': opp_max,
        'opp_count': float(opp_count),
    }

    # 窗口特征
    for w in [30, 60, 120]:
        mask = (all_times >= bait_time - w) & (all_times < bait_time)
        count = mask.sum()
        feat_values[f'past_{w}s_count'] = float(count)
        if count > 0:
            feat_values[f'past_{w}s_avg'] = float(all_scores[mask].mean())
            feat_values[f'past_{w}s_max'] = float(all_scores[mask].max())

    return feat_values


# ============================
# 主推理函数
# ============================
def run_inference(excel_path, model_path='best_model.pkl',
                  scaler_path='scaler.pkl', features_path='feature_names.json',
                  start_pos=(5, 5), output_excel=None, use_test_only=True):
    """
    读取食饵数据表，逐次输出机器人前进方向。

    与 solve.py 中 simulate_online 逻辑一致：
    - 按时间顺序处理每个食饵
    - 机器人以1m/s速度移动，考虑曼哈顿距离
    - MLP模型预测追/跳
    - 追捕时机器人移动到食饵位置（耗时 = 曼哈顿距离）

    参数:
        output_excel: 若指定路径，则保存结果到Excel文件
        use_test_only: True=仅用后50%测试集, False=用全部数据
    """

    # 加载模型
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_names = json.load(f)

    # 加载数据
    df = pd.read_excel(excel_path)
    df.columns = ['time', 'position', 'score']
    df['pos'] = df['position'].apply(parse_position)
    df = df.sort_values('time').reset_index(drop=True)

    # ★ 默认取后50%作为测试集（与训练时划分一致）
    if use_test_only:
        split_idx = len(df) // 2
        df = df.iloc[split_idx:].copy().reset_index(drop=True)
        print(f"使用测试集（后50%）: {len(df)} 条")

    # 归一化时间从0开始
    times = df['time'].values.astype(np.float64)
    times = times - times[0]
    positions = list(df['pos'].values)
    scores = df['score'].values.astype(np.float64)
    total_baits = len(df)

    # 状态
    robot_pos = start_pos
    robot_time = 0.0
    total_score = 0
    results = []

    print("=" * 85)
    print(f"在线推理 | 起始位置: {start_pos} | 食饵总数: {total_baits}")
    print("=" * 85)
    header = (f"{'序号':>4s}  {'时刻':>7s}  {'食饵':>8s}  {'分值':>4s}  "
              f"{'机器人':>8s}  {'距离':>4s}  {'决策':>6s}  {'方向':>8s}  {'得分':>6s}")
    print(header)
    print("-" * 85)

    for i in range(total_baits):
        bait_time = times[i]
        bait_pos = positions[i]
        bait_score = int(scores[i])

        # 距离与可达性
        dist = manhattan(robot_pos, bait_pos)
        arrival_time = robot_time + dist
        can_reach = arrival_time <= bait_time + BAIT_LIFETIME

        decision = 'skip'
        direction = (0, 0)
        robot_pos_before = robot_pos  # ★ 保存决策前位置

        if can_reach:
            # 构建特征
            feat_values = build_features(
                robot_pos, robot_time, bait_time, bait_pos, bait_score,
                i, times, scores, total_baits
            )
            feat_vec = np.array([[feat_values.get(c, 0.0) for c in feature_names]],
                                dtype=np.float64)
            feat_scaled = scaler.transform(feat_vec)
            pred = model.predict(feat_scaled)[0]

            if pred == 1:
                decision = 'pursue'
                # 方向：从 robot_pos 到 bait_pos 的第一步
                direction = compute_direction(robot_pos, bait_pos)

                # 移动机器人
                robot_pos = bait_pos
                arrival_at_bait = max(arrival_time, bait_time)
                robot_time = arrival_at_bait
                total_score += bait_score

        # 记录
        results.append({
            'index': i + 1,
            'time': bait_time,
            'bait_pos': bait_pos,
            'bait_score': bait_score,
            'robot_pos_before': robot_pos_before,
            'robot_pos_after': robot_pos,
            'dist': dist,
            'decision': decision,
            'direction': direction,
            'total_score': total_score,
        })

        # 打印
        d = direction
        dir_str = f"({d[0]:+d},{d[1]:+d})"
        print(f"{i+1:>4d}  {bait_time:>7.1f}s  ({bait_pos[0]:>2d},{bait_pos[1]:>2d})  "
              f"{bait_score:>4d}  ({robot_pos[0]:>2d},{robot_pos[1]:>2d})  "
              f"{dist:>4d}  {decision:>6s}  {dir_str:>8s}  {total_score:>6d}")

    print("-" * 85)
    pursue_count = sum(1 for r in results if r['decision'] == 'pursue')
    skip_count = sum(1 for r in results if r['decision'] == 'skip')
    print(f"追捕: {pursue_count}  跳过: {skip_count}  总得分: {total_score}")
    print(f"总食饵分: {int(scores.sum())}  得分率: {total_score/scores.sum()*100:.1f}%")

    # ---- 导出 Excel ----
    if output_excel:
        save_to_excel(results, output_excel, start_pos, total_score, int(scores.sum()))

    return results


def save_to_excel(results, output_path, start_pos, total_score, total_bait_score):
    """将推理结果保存为 Excel 文件"""
    rows = []
    for r in results:
        d = r['direction']
        bp = r['bait_pos']
        rb = r['robot_pos_before']
        ra = r['robot_pos_after']
        dist = r['dist']

        rows.append({
            '序号': r['index'],
            '出现时刻(s)': r['time'],
            '食饵行': bp[0],
            '食饵列': bp[1],
            '食饵位置': f"({bp[0]},{bp[1]})",
            '分值': r['bait_score'],
            '机器人行(决策前)': rb[0],
            '机器人列(决策前)': rb[1],
            '机器人位置(决策前)': f"({rb[0]},{rb[1]})",
            '曼哈顿距离': dist,
            '可达(距离≤3)': '是' if dist <= 3 else '否',
            '模型决策': '追捕' if r['decision'] == 'pursue' else '跳过',
            'X方向(左右)': d[0],
            'Y方向(上下)': d[1],
            '方向向量': f"({d[0]:+d},{d[1]:+d})",
            '方向说明': direction_desc(d),
            '机器人行(决策后)': ra[0],
            '机器人列(决策后)': ra[1],
            '机器人位置(决策后)': f"({ra[0]},{ra[1]})",
            '累计得分': r['total_score'],
        })

    df_out = pd.DataFrame(rows)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 详细结果
        df_out.to_excel(writer, sheet_name='推理结果', index=False)

        # Sheet 2: 汇总统计
        pursue = sum(1 for r in results if r['decision'] == 'pursue')
        skip = sum(1 for r in results if r['decision'] == 'skip')
        summary = pd.DataFrame([
            ['起始位置', f"({start_pos[0]}, {start_pos[1]})"],
            ['食饵总数', len(results)],
            ['追捕次数', pursue],
            ['跳过次数', skip],
            ['追捕比例', f"{pursue/len(results)*100:.1f}%"],
            ['总得分', total_score],
            ['总食饵分值', total_bait_score],
            ['得分率', f"{total_score/total_bait_score*100:.1f}%"],
            ['模型', 'MLP (128,64,32)'],
        ], columns=['指标', '值'])
        summary.to_excel(writer, sheet_name='汇总统计', index=False)

    print(f"\n结果已保存到: {output_path}")


def direction_desc(d):
    """方向向量的中文描述"""
    mapping = {
        (1, 0): '右',
        (-1, 0): '左',
        (0, 1): '上',
        (0, -1): '下',
        (0, 0): '不动',
    }
    return mapping.get(d, str(d))


def output_directions_only(results):
    """仅输出方向向量序列"""
    print("\n" + "=" * 50)
    print("方向向量序列 (右=(1,0) 左=(-1,0) 上=(0,1) 下=(0,-1) 不动=(0,0)):")
    print("=" * 50)
    for r in results:
        d = r['direction']
        print(f"  #{r['index']:>4d}  t={r['time']:>7.1f}s  "
              f"食饵({r['bait_pos'][0]:>2d},{r['bait_pos'][1]:>2d}) "
              f"分{r['bait_score']:>3d}  "
              f"{r['decision']:>6s}  → ({d[0]:+d},{d[1]:+d})")


# ============================
# 命令行入口
# ============================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='机器人竞赛 - 在线推理')
    parser.add_argument('input', nargs='?',
                        default='202607032000建模复赛题附件.xlsx',
                        help='食饵数据Excel文件')
    parser.add_argument('--start-row', type=int, default=5, help='起始行 1-10 (默认5)')
    parser.add_argument('--start-col', type=int, default=5, help='起始列 1-10 (默认5)')
    parser.add_argument('--model', default='best_model.pkl', help='模型文件')
    parser.add_argument('--output', '-o', default='inference_result.xlsx',
                        help='输出Excel文件路径 (默认 inference_result.xlsx)')
    parser.add_argument('--all-data', action='store_true',
                        help='使用全部数据（默认仅用后50%测试集）')
    parser.add_argument('--directions-only', action='store_true', help='仅输出方向序列')

    args = parser.parse_args()
    start_pos = (args.start_row, args.start_col)

    results = run_inference(
        args.input, args.model,
        'scaler.pkl', 'feature_names.json',
        start_pos,
        output_excel=args.output,
        use_test_only=not args.all_data
    )

    if args.directions_only:
        output_directions_only(results)
