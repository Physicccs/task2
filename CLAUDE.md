# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This directory currently contains no source code — only the problem statement and
dataset for a math-modeling competition. There is no build, lint, or test tooling to
document yet. If code is added later (e.g. a Python project to analyze the dataset and
simulate the strategy), update this file with real commands and architecture notes at
that point.

## Contents

- `202607032000建模复赛题.pdf` — competition prompt: "机器人场地竞赛的策略" (Robot
  Arena Competition Strategy). A small robot moves at 1 m/s on a 9m×9m square arena
  with a 10×10 grid (1m spacing). Scored "bait" appears at random grid intersections,
  stays for 3 seconds, then disappears. The robot only learns a bait's position/value
  when it appears, and cannot predict future bait. The robot can only move along grid
  lines. Required tasks:
  1. Analyze the pattern of bait appearances from the attached dataset.
  2. Design a strategy for the robot to maximize score.
  3. Estimate the strategy's average points per minute.
  4. Discuss the strategy's applicability to real-world robot arena scenarios.
  - Submission deadline: **2026-07-06 (Monday) 20:00**, via
    https://sjmma.bearshen.com/participant/access/8Lyf9QQ6UbjVCt9lmj10
  - Report must be PDF, ≤21 pages total (including abstract; appendix/references
    exempt). Any code written may be included in the appendix (not counted toward the
    21-page limit). Do not submit software or executables.

- `202607032000建模复赛题附件.xlsx` — dataset of bait appearances over a 3-hour
  window. `Sheet1` columns:
  - `出现时刻` — appearance time, in seconds
  - `位置` — grid position, e.g. `(3,1)` (x, y ∈ [1,10])
  - `分值` — point value of the bait
  - `Sheet2` is empty.

Note: both filenames are GBK-encoded on disk; tools that assume UTF-8 filenames (e.g.
directly opening by name) may fail to resolve them and may need the raw byte path or a
renamed copy.
