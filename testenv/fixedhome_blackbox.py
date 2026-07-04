"""Black-box strategy: hard-pinned home cell + greedy ("固定常驻格+贪心").

Same greedy approach as strategy_blackbox.py (chase the best reachable
bait -- highest value; ties by distance, then by undiscounted path
control-power -- and greedy-step toward home when idle; the control map
is still built from the training set and updated online exactly as
before, so path-score tie-breaks keep using live statistics), except the
idle target is hard-pinned at (3, 7) -- the true global-optimum cell
found from the full dataset throughout this analysis -- instead of being
recomputed as the argmax of the (training-set-derived, then online-
updated) control map every time new data arrives.

This isolates one specific question: does letting the home cell move
(and occasionally flicker between near-tied candidates, as
strategy_blackbox.py does) cost anything relative to just committing to
the best-known point and never touching it again?

Training (the control map used only for path-score tie-breaks) is built
from the first --train-events records (default 893); the home cell itself
is a constant regardless of --train-events.

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/fixedhome_blackbox.py" \
      --skip-events 893 --out replay_fixedhome.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate_strategy as strat

HOME = (3, 7)


def decide_move(pos, active, t, control):
    reachable = [b for b in active
                 if strat.manhattan(pos, b['pos']) <= b['expire'] - t]
    if reachable:
        reachable.sort(key=lambda b: (-b['val'], strat.manhattan(pos, b['pos']),
                                       -strat.path_score(pos, b['pos'], control)))
        return strat.step_towards(pos, reachable[0]['pos'])
    return strat.greedy_step(pos, HOME, control)


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'))
    ap.add_argument('--train-events', type=int, default=893)
    args = ap.parse_args()

    train = strat.load_events(args.data)[:args.train_events]
    counts = {c: 0 for c in strat.CELLS}
    for _, p, _ in train:
        counts[p] += 1
    total = len(train)
    control = strat.control_map(counts, total)
    print(f'[fixedhome_blackbox] home fixed at {HOME} (never recomputed)',
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
            p = (int(x), int(y))
            new.append(p)
            active.append({'pos': p, 'val': float(v), 'expire': t + strat.BAIT_LIFE})
        if new:
            for p in new:
                counts[p] += 1
                total += 1
            control = strat.control_map(counts, total)
        active = [b for b in active if b['expire'] >= t and b['pos'] != pos]

        nxt = decide_move(pos, active, t, control)
        dx, dy = nxt[0]-pos[0], nxt[1]-pos[1]
        pos = nxt
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
