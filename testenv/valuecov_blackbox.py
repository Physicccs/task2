"""Black-box strategy: value-weighted coverage + greedy ("价值覆盖+贪心").

Same greedy approach as strategy_blackbox.py (chase the best reachable
bait -- highest value; ties by distance, then by undiscounted path
control-power -- and greedy-step toward the current best cell when idle),
but the spatial coverage weight for each cell is the *total value* of
baits that have appeared there, not the occurrence count:

    control(cell) = sum over c2 within Manhattan-3 of value_sum(c2)

instead of simulate_strategy.control_map's sum of occurrence probability.
This is the alpha=1 (pure value-weighted) endpoint of evweight_blackbox.py's
tunable blend -- kept here as its own fixed, dedicated strategy rather than
something a grid search can tune away, so it can be compared directly
against the occurrence-only baseline.

The model (value_sum per cell) is built from the first --train-events
records (default 893), then updated online as test baits arrive, exactly
like strategy_blackbox.py's control map.

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/valuecov_blackbox.py" \
      --skip-events 893 --out replay_valuecov.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate_strategy as strat


def value_control_map(value_sum):
    return {c: sum(value_sum[c2] for c2 in strat.NEIGHBORS[c]) for c in strat.CELLS}


def decide_move(pos, active, t, control):
    reachable = [b for b in active
                 if strat.manhattan(pos, b['pos']) <= b['expire'] - t]
    if reachable:
        reachable.sort(key=lambda b: (-b['val'], strat.manhattan(pos, b['pos']),
                                       -strat.path_score(pos, b['pos'], control)))
        return strat.step_towards(pos, reachable[0]['pos'])
    home = max(control, key=control.get)
    return strat.greedy_step(pos, home, control)


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'))
    ap.add_argument('--train-events', type=int, default=893)
    args = ap.parse_args()

    train = strat.load_events(args.data)[:args.train_events]
    value_sum = {c: 0.0 for c in strat.CELLS}
    for _, p, v in train:
        value_sum[p] += v
    control = value_control_map(value_sum)
    home = max(control, key=control.get)
    print(f'[valuecov_blackbox] home={home} (value-weighted control={control[home]:.1f})',
          file=sys.stderr)

    pos = (1, 1)
    active = []
    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == 'INIT':
            continue
        if parts[0] == 'END':
            break

        t, k = int(parts[1]), int(parts[2])
        new = []
        for _ in range(k):
            x, y, v = sys.stdin.readline().split()
            p, val = (int(x), int(y)), float(v)
            new.append((p, val))
            active.append({'pos': p, 'val': val, 'expire': t + strat.BAIT_LIFE})
        if new:
            for p, v in new:
                value_sum[p] += v
            control = value_control_map(value_sum)
        active = [b for b in active if b['expire'] >= t and b['pos'] != pos]

        nxt = decide_move(pos, active, t, control)
        dx, dy = nxt[0]-pos[0], nxt[1]-pos[1]
        pos = nxt
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
