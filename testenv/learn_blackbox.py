"""Black-box adapter for the strategies in learn.py.

Speaks the test-environment protocol of testenv/env.py on stdin/stdout and
delegates every per-second decision to one of learn.py's strategy functions:

  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/learn_blackbox.py --strategy G" \
      --skip-events 893 --out replay_learn.html

Strategies: A static-best-position, B greedy-value, C value-per-distance,
D adaptive-wait, E ml-value-map, F weighted (tune with --alpha/--beta),
G time-aware, H patrol, I adaptive-v2.

Faithfulness: the decision loop mirrors RobotSimulator.simulate_discrete()
exactly — same active-food list contents (including the food being eaten
this very second, which learn.py's simulator only removes on the next
tick), same one-step-per-second movement resolving rows before columns.

Legality guarantees required by the environment:
  - learn.py strategies return a TARGET cell in 0-indexed (r, c); this
    adapter converts it into a single step and emits (dx, dy), always one
    of (0,0)/(±1,0)/(0,±1), clamped to the board, as plain ints.
  - learn.py's (r, c) maps to the protocol's (x, y) as r = x-1, c = y-1
    (r is the first number of the dataset position string).
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# learn.py prints its data-loading summary at import time; keep the
# protocol stdout clean by silencing it.
with open(os.devnull, 'w') as _devnull, contextlib.redirect_stdout(_devnull):
    import learn

GRID = learn.GRID_SIZE
LIFE = learn.FOOD_LIFETIME
ALLOWED = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}


def build_strategies(alpha, beta):
    return {
        'A': learn.strategy_static_best,
        'B': learn.strategy_greedy_value,
        'C': learn.strategy_value_per_distance,
        'D': learn.strategy_adaptive_wait,
        'E': learn.strategy_ml_optimal,
        'F': learn.make_weighted_strategy(alpha, beta),
        'G': learn.strategy_time_aware,
        'H': learn.strategy_patrol,
        'I': learn.strategy_adaptive_v2,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--strategy', default='G', choices=list('ABCDEFGHI'),
                    help='which learn.py strategy to run (default G)')
    ap.add_argument('--alpha', type=float, default=1.0,
                    help='value-weight exponent for strategy F')
    ap.add_argument('--beta', type=float, default=0.5,
                    help='distance-penalty constant for strategy F')
    args = ap.parse_args()
    strategy = build_strategies(args.alpha, args.beta)[args.strategy]

    r, c = 0, 0                 # learn.py coords; env start (1,1) = (0,0)
    foods = []                  # (t_appear, r, c, value)
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

        # mirror simulate_discrete: build the active list first, then mark
        # (at most) the one food under the robot as eaten — the strategy
        # still sees it in `active` for this tick
        active = []
        for i, (f_t, f_r, f_c, f_v) in enumerate(foods):
            if i in eaten:
                continue
            if f_t <= t < f_t + LIFE:
                active.append((i, f_r, f_c, f_v, LIFE - (t - f_t)))
        for i, f_r, f_c, f_v, rem in active:
            if (f_r, f_c) == (r, c):
                eaten.add(i)
                break

        target = strategy(r, c, active, t)
        tr, tc = (r, c) if target is None else (int(target[0]), int(target[1]))

        nr, nc = r, c
        if nr < tr:
            nr += 1
        elif nr > tr:
            nr -= 1
        elif nc < tc:
            nc += 1
        elif nc > tc:
            nc -= 1

        # legality guard: clamp to the board, force one of the 5 moves
        nr = max(0, min(GRID - 1, nr))
        nc = max(0, min(GRID - 1, nc))
        dr, dc = nr - r, nc - c
        if (dr, dc) not in ALLOWED:
            dr, dc = 0, 0
            nr, nc = r, c
        r, c = nr, nc

        # (r, c) -> (x, y): dx = dr, dy = dc
        sys.stdout.write(f'{dr} {dc}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
