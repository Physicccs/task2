"""Black-box strategy: strategy_time_aware (G) with ZERO training data.

Diagnostic baseline for "so how does G perform with no training data?":
block_best_positions collapses to a single default point (0,0) in 0-indexed
r,c -- i.e. (1,1) in the protocol's 1-indexed x,y, exactly the robot's own
start cell -- because there is no data to compute any other statistic
from. G's decision formula itself (score = value - 0.1*future_dist) is
otherwise unchanged; future_dist is just always measured against that same
default point since there is only one "time block".

This isolates how much of G's performance actually comes from its
block_best_positions statistics versus from the reachable-greedy chase
logic alone.

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/g_notrain_blackbox.py" \
      --skip-events 893 --out replay_g_notrain.html
"""

import sys

GRID_SIZE = 10
FOOD_LIFETIME = 3
DEFAULT_POS = (0, 0)   # 0-indexed r,c == protocol (1,1), the robot's own start
ALLOWED = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}


def strategy_time_aware_notrain(robot_r, robot_c, active, t):
    wr, wc = DEFAULT_POS
    if not active:
        return (wr, wc)
    best, best_score = None, -1
    for idx, f_r, f_c, f_v, rem in active:
        d = abs(robot_r - f_r) + abs(robot_c - f_c)
        if d <= rem:
            future_dist = abs(f_r - wr) + abs(f_c - wc)
            score = f_v - 0.1 * future_dist
            if score > best_score:
                best_score = score
                best = (f_r, f_c)
    return best if best else (wr, wc)


def main():
    r, c = 0, 0
    tr, tc = 0, 0
    foods = []
    eaten = set()

    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == 'INIT':
            continue
        if parts[0] == 'END':
            break

        t, k = int(parts[1]), int(parts[2])
        for _ in range(k):
            x, y, v = sys.stdin.readline().split()
            foods.append((t, int(x) - 1, int(y) - 1, float(v)))

        active = []
        for i, (f_t, f_r, f_c, f_v) in enumerate(foods):
            if i in eaten:
                continue
            if f_t <= t < f_t + FOOD_LIFETIME:
                active.append((i, f_r, f_c, f_v, FOOD_LIFETIME - (t - f_t)))
        for i, f_r, f_c, f_v, rem in active:
            if (f_r, f_c) == (r, c):
                eaten.add(i)

        target = strategy_time_aware_notrain(r, c, active, t)
        if target is not None:
            tr, tc = int(target[0]), int(target[1])

        nr, nc = r, c
        if nr < tr:
            nr += 1
        elif nr > tr:
            nr -= 1
        elif nc < tc:
            nc += 1
        elif nc > tc:
            nc -= 1

        nr = max(0, min(GRID_SIZE - 1, nr))
        nc = max(0, min(GRID_SIZE - 1, nc))
        dr, dc = nr - r, nc - c
        if (dr, dc) not in ALLOWED:
            dr, dc = 0, 0
            nr, nc = r, c
        r, c = nr, nc

        sys.stdout.write(f'{dr} {dc}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
