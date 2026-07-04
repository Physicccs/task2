"""Black-box adapter for the strategy in simulate_strategy.py.

Wraps the original strategy (kept untouched) in the test-environment
stdin/stdout protocol of testenv/env.py, so it can be tested and
visualized like any other black box:

  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/strategy_blackbox.py --T 0" \
      --skip-events 893 --start-pos 3,8 --out replay.html

Faithful to simulate_train_test(): the initial control map is built from
the first --train-events records of the dataset, then updated online as
test baits arrive; the robot chases the best reachable bait ranked by
(-value, distance, -path_score) and drifts toward the control-map "home"
cell when idle (path_score is the plain, undiscounted control-power sum
along the shortest path -- see simulate_strategy.py). A bait appearing at
ta is catchable up to the instant ta+3, so it is worth chasing when
manhattan(pos, bait) <= expire - t, matching the original simulator.

The robot starts at its own computed home cell, not the environment's
default (1,1) corner -- this models a robot given free preparation time
before the match to walk to a self-chosen cell (see report's starting-
position assumption). Pass --start-pos to testenv/env.py matching that
same home cell so the environment's own position bookkeeping stays in
sync with this process's internal `pos`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate_strategy as strat


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'),
                    help='xlsx dataset used to build the initial model')
    ap.add_argument('--train-events', type=int, default=893,
                    help='number of leading records used as training data')
    ap.add_argument('--T', type=float, default=0,
                    help='ignore baits with value below this threshold')
    args = ap.parse_args()

    train = strat.load_events(args.data)[:args.train_events]
    counts = {c: 0 for c in strat.CELLS}
    for _, p, v in train:
        if v >= args.T:
            counts[p] += 1
    total = sum(counts.values())
    control = strat.control_map(counts, total)
    home = max(control, key=control.get)

    pos = home                  # free prep time: start already at the trained home cell
    active = []                 # {'pos', 'val', 'expire'}

    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == 'INIT':
            continue
        if parts[0] == 'END':
            break

        t, k = int(parts[1]), int(parts[2])
        for _ in range(k):
            x, y, v = sys.stdin.readline().split()
            b = {'pos': (int(x), int(y)), 'val': float(v),
                 'expire': t + strat.BAIT_LIFE}
            active.append(b)
            if b['val'] >= args.T:
                counts[b['pos']] += 1
                total += 1
        if k:
            control = strat.control_map(counts, total)
            home = max(control, key=control.get)

        # We stand at pos at instant t; the env has already collected any
        # bait under us. A move lands at t+1 and baits are catchable up to
        # the instant `expire`, so keep only baits still catchable then.
        active = [b for b in active
                  if b['expire'] > t and b['pos'] != pos]

        reachable = [b for b in active
                     if b['val'] >= args.T
                     and strat.manhattan(pos, b['pos']) <= b['expire'] - t]

        if reachable:
            reachable.sort(key=lambda b: (
                -b['val'],
                strat.manhattan(pos, b['pos']),
                -strat.path_score(pos, b['pos'], control)))
            nxt = strat.step_towards(pos, reachable[0]['pos'])
        else:
            nxt = strat.greedy_step(pos, home, control)

        dx, dy = nxt[0] - pos[0], nxt[1] - pos[1]
        pos = nxt
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
