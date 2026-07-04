"""RobotPolicy for the arena competition, conforming to new_standard.md.

Strategy (control-power patrol + greedy value chase), trained only on the
first 893 chronological bait records of the provided dataset:

  1. Spatial prior: for each grid cell c, control(c) = sum of p(c') over all
     c' within Manhattan distance <= 3 of c, where p is the empirical bait
     appearance probability. The "home" cell is argmax(control) -- the cell
     from which the most future bait is reachable within its 3s lifetime.
  2. When bait is on the field and reachable in time, chase the best one,
     ranked by (highest value, shortest distance, highest control power
     along the path as a tie-breaker).
  3. When idle, drift one step toward home, preferring whichever of the (at
     most two) distance-reducing neighbor cells has higher control power.
  4. The prior is trained once from train_counts.json (first 893 events
     only) and then updated online as new baits are observed during a live
     run, exactly as validated in this project's black-box evaluation.

This file has no dependency on the rest of the project -- only the
shipped weights file next to it.
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def _find_weights():
    """按优先级定位 train_counts.json：同目录（注释所述 shipped 布局）优先，
    再退回 your_model_weights/ 子目录。避免打包路径不一致导致加载失败。"""
    for p in (os.path.join(BASE, 'train_counts.json'),
              os.path.join(BASE, 'your_model_weights', 'train_counts.json')):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        'train_counts.json 未找到（已查同目录与 your_model_weights/ 子目录）')

WEIGHTS_FILE = _find_weights()

GRID = 10
RADIUS = 3
BAIT_LIFE = 3

CELLS = [(x, y) for x in range(1, GRID + 1) for y in range(1, GRID + 1)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


NEIGHBORS = {c: [c2 for c2 in CELLS if manhattan(c, c2) <= RADIUS] for c in CELLS}


def control_map(counts, total):
    if total == 0:
        prob = {c: 1.0 / len(CELLS) for c in CELLS}
    else:
        prob = {c: counts[c] / total for c in CELLS}
    return {c: sum(prob[c2] for c2 in NEIGHBORS[c]) for c in CELLS}


def path_score(a, b, control):
    """Best achievable sum of control power over any monotone shortest path
    from a to b; used only to break ties among equally-good chase targets."""
    if a == b:
        return control[a]
    dx, dy = b[0] - a[0], b[1] - a[1]
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    nx, ny = abs(dx), abs(dy)

    def cell(i, j):
        return (a[0] + i * sx, a[1] + j * sy)

    g = [[0.0] * (ny + 1) for _ in range(nx + 1)]
    for i in range(nx, -1, -1):
        for j in range(ny, -1, -1):
            best = None
            if i < nx:
                best = g[i + 1][j]
            if j < ny:
                best = g[i][j + 1] if best is None else max(best, g[i][j + 1])
            g[i][j] = control[cell(i, j)] + (best or 0.0)
    return g[0][0]


def step_towards(cur, target):
    x, y = cur
    tx, ty = target
    if x != tx:
        x += 1 if tx > x else -1
    elif y != ty:
        y += 1 if ty > y else -1
    return (x, y)


def greedy_step(start, goal, control):
    """Idle move: among the (<=2) neighbor cells that make monotone progress
    toward goal, step to whichever has higher control power."""
    if start == goal:
        return start
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    candidates = []
    if dx != 0:
        candidates.append((start[0] + (1 if dx > 0 else -1), start[1]))
    if dy != 0:
        candidates.append((start[0], start[1] + (1 if dy > 0 else -1)))
    return max(candidates, key=lambda c: control[c])


class RobotPolicy:
    def __init__(self):
        with open(WEIGHTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.counts = {c: 0 for c in CELLS}
        for key, n in data['counts'].items():
            x, y = key.split(',')
            self.counts[(int(x), int(y))] = n
        self.total = data['total']

        self.control = control_map(self.counts, self.total)
        self.home = max(self.control, key=self.control.get)

        self._seen = set()   # (ta, x, y) keys already folded into counts

    def act(self, t, active, pos=None):
        if pos is None:
            pos = self.home
        pos = (int(round(pos[0])), int(round(pos[1])))

        new_seen = False
        for ta, x, y, v in active:
            key = (ta, x, y)
            if key not in self._seen:
                self._seen.add(key)
                self.counts[(x, y)] += 1
                self.total += 1
                new_seen = True
        if new_seen:
            self.control = control_map(self.counts, self.total)
            self.home = max(self.control, key=self.control.get)

        reachable = [
            (ta, x, y, v) for ta, x, y, v in active
            if manhattan(pos, (x, y)) <= (ta + BAIT_LIFE - t)
        ]

        if reachable:
            reachable.sort(key=lambda b: (
                -b[3],
                manhattan(pos, (b[1], b[2])),
                -path_score(pos, (b[1], b[2]), self.control)))
            target = (reachable[0][1], reachable[0][2])
            nxt = step_towards(pos, target)
        else:
            nxt = greedy_step(pos, self.home, self.control)

        return (nxt[0] - pos[0], nxt[1] - pos[1])
