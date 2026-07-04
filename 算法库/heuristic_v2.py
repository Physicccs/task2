# -*- coding: utf-8 -*-
"""
heuristic_v2 —— 手工拦截策略 v2（统一赛马接口，零第三方依赖）
====================================================================
统一接口：RobotPolicy.act(self, t, active, pos=None) → (dx, dy)
- t: 当前秒（整数）；active: 在场食饵 [(出现时刻, x, y, v), ...]，表格坐标 1..10
- 返回 (0,0)/(±1,0)/(0,±1)，每秒调用一次
挂载：D:/conda/envs/AI/python.exe task2/pk_task2.py --gui  →「添加模型…」选本文件
     或 --policy task2/heuristic_v2.py

【与 v1（robot_policy.py）的唯一差别：jiggle 死区兜底】
整秒判定器（"位置相等即得分"，pk_task2 现行口径）下脚下食饵在调用 act 前已被
拾取，jiggle 不触发——v2 与 v1 严格等价（测试段 894 起均为 2854 分 / 31.55 分/分钟）。
但若评测器只按"扫过的线段"判拾（连续模拟器常见写法），v1 对脚下食饵返回零动作
会静止漏拾（实验实测损失 23 枚 / 175 分）；v2 此时朝场地中心横跨一步，扫段经过
本交点即拾取。故 v2 对任意判定口径鲁棒，对外发布/赛马一律用 v2。

【策略内容（继承 v1，经统计结论驱动设计 + 优化探讨证实已达局部最优）】
1. 空闲驻守覆盖率最优等待点 (3,7)（曼哈顿 3m 覆盖 39.5% 经验落点）
2. 可达食饵带截止期排列调度（价值前 5 全排列取总价值最大、并列取完成最早）
3. 拾取后立即回撤等待点
优化探讨结论（详见 output/任务2_手工策略优化探讨报告.pdf 与 task2/variants/README.md）：
逐饵归因软错过=0，等待点/追击门槛/终点感知调度/路径塑形/双驻点驻留全部被实验
证否，31.55 分/分钟已逼近由"落点不可预测"决定的在线天花板（全知 oracle 74.8 不可及）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_policy import RobotPolicy as _V1RobotPolicy       # 零依赖整秒版 v1


class RobotPolicy(_V1RobotPolicy):
    """统一赛马接口版 v2：v1 整秒策略 + 脚下死区 jiggle 兜底。零第三方依赖。"""

    def act(self, t, active, pos=None):
        if pos is not None:
            self.pos = tuple(pos)
        tgt = self._best_first_target(t, active)
        if tgt is not None and tgt == self.pos:
            # 首目标就在脚下：只认"扫段"的判定器拾不到静止目标 → 朝场地中心
            # 横跨一步，扫段经过本交点即拾取；整秒判定器下此分支不会触发。
            step = (1, 0) if self.pos[0] < 5.5 else (-1, 0)
            self.pos = (self.pos[0] + step[0], self.pos[1] + step[1])
            return step
        return super().act(t, active, pos=None)      # pos 已同步，不重复传
