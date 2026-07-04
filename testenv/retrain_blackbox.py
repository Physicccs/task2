"""Black-box adapter for the ML policy trained by retrain_model.py.

Speaks the test-environment protocol of testenv/env.py on stdin/stdout:

  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/retrain_blackbox.py --strategy hybrid" \
      --skip-events 893 --out replay_retrain.html

Strategies:
  hybrid  - chase the best reachable bait greedily, ask the ML model for
            the move when nothing is reachable (default)
  pure    - every move comes from the ML model
  teacher - strategy_time_aware, the behavior-cloning teacher (reference)

The policy bundle is loaded from robot_policy_model.pkl; if it does not
exist yet, it is trained first via retrain_model.retrain() (progress goes
to stderr so the protocol stdout stays clean).

Faithfulness: the decision loop mirrors retrain_model._run_sim() — the
active list uses the same window/remaining-time bookkeeping, every food
under the robot is marked eaten, and movement resolves one step per
second, rows before columns.

Legality guarantees required by the environment:
  - strategies return a TARGET cell in 0-indexed (r, c); the adapter turns
    it into a single step and always emits one of (0,0)/(±1,0)/(0,±1) as
    plain ints, clamped to the board.
  - (r, c) maps to the protocol's (x, y) as r = x-1, c = y-1.
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
with open(os.devnull, 'w') as _devnull, contextlib.redirect_stdout(_devnull):
    import retrain_model as rm

GRID = rm.GRID_SIZE
LIFE = rm.FOOD_LIFETIME
ALLOWED = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}


def get_strategy(name, model_path):
    if name == 'teacher':
        return rm.strategy_time_aware
    if not os.path.exists(model_path):
        print(f'[retrain_blackbox] {model_path} not found, training now...',
              file=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):
            model_data = rm.retrain(model_path)
    else:
        model_data = rm.load_policy(model_path)
    hybrid, pure = rm.make_ml_strategies(model_data)
    return hybrid if name == 'hybrid' else pure


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--strategy', default='hybrid',
                    choices=['hybrid', 'pure', 'teacher'])
    ap.add_argument('--model', default=rm.MODEL_PATH,
                    help='path of the pickled policy bundle')
    args = ap.parse_args()
    strategy = get_strategy(args.strategy, args.model)

    r, c = 0, 0                 # retrain_model coords; env (1,1) = (0,0)
    tr, tc = 0, 0               # current target, kept when strategy is None
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

        # mirror _run_sim: active list, then mark every food under the
        # robot as eaten (the env has just collected them)
        active = []
        for i, (f_t, f_r, f_c, f_v) in enumerate(foods):
            if i in eaten:
                continue
            if f_t <= t < f_t + LIFE:
                active.append((i, f_r, f_c, f_v, LIFE - (t - f_t)))
        for i, f_r, f_c, f_v, rem in active:
            if (f_r, f_c) == (r, c):
                eaten.add(i)

        target = strategy(r, c, active, t)
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
