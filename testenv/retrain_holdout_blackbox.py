"""Black-box adapter for the training-set-only ML policy in
retrain_model_holdout.py (all statistics and behavior-cloning built from
events[:893] alone -- a genuine train/test split, unlike retrain_blackbox.py
whose underlying retrain_model.py always uses the full dataset).

Speaks the test-environment protocol of testenv/env.py on stdin/stdout:

  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/retrain_holdout_blackbox.py --strategy hybrid" \
      --skip-events 893 --out replay_retrain_holdout.html

Strategies: hybrid (default), pure, teacher -- see retrain_model_holdout.py.

Faithfulness note: the model's time features are relative to its training
window's own start (t - T_START, see retrain_model_holdout.TOTAL_TIME).
Since the live protocol feeds absolute dataset time (which for the test
set starts around t=5376, not 0), this adapter re-bases every incoming
t to be relative to the first t it sees, so the model is queried with the
same kind of "seconds since my operating window started" value it was
trained on.
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
with open(os.devnull, 'w') as _devnull, contextlib.redirect_stdout(_devnull):
    import retrain_model_holdout as rm

GRID = rm.GRID_SIZE
LIFE = rm.FOOD_LIFETIME
ALLOWED = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}


def get_strategy(name, model_path):
    if name == 'teacher':
        return rm.strategy_time_aware
    if not os.path.exists(model_path):
        print(f'[retrain_holdout_blackbox] {model_path} not found, training now...',
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
    ap.add_argument('--model', default=rm.MODEL_PATH)
    args = ap.parse_args()
    strategy = get_strategy(args.strategy, args.model)

    r, c = 0, 0
    tr, tc = 0, 0
    foods = []                  # (t_appear_rel, r, c, value)
    eaten = set()
    t_base = None                # first absolute t seen -> rebased to 0

    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == 'INIT':
            continue
        if parts[0] == 'END':
            break

        t_abs, k = int(parts[1]), int(parts[2])
        if t_base is None:
            t_base = t_abs
        t = t_abs - t_base

        for _ in range(k):
            x, y, v = sys.stdin.readline().split()
            foods.append((t, int(x) - 1, int(y) - 1, float(v)))

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

        nr = max(0, min(GRID - 1, nr))
        nc = max(0, min(GRID - 1, nc))
        dr, dc = nr - r, nc - c
        if (dr, dc) not in ALLOWED:
            dr, dc = 0, 0
            nr, nc = r, c
        r, c = nr, nc

        sys.stdout.write(f'{dr} {dc}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
