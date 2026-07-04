"""Black-box strategy: recency-weighted heat tracking + greedy ("近期热度追踪").

Idea: instead of a static, all-time cumulative probability map, track a
per-cell "heat" that gets +1 every time a bait appears there and decays by
a fixed multiplier every second, so recent appearances count for more than
old ones. The idle "home" is recomputed continuously as whichever cell has
the highest heat-derived control power (sum of neighbor heat within the
usual Manhattan-3 range); the robot greedy-steps toward it exactly like
simulate_strategy.greedy_step. Chasing logic is unchanged: greedily go for
the best reachable bait (highest value; ties by distance, then by
undiscounted path control-power computed on the current heat map).

The only free parameter is the heat half-life (seconds). It is trained by
grid search over a candidate list -- including a near-infinite half-life,
which degenerates to the static all-time model as a built-in baseline --
replaying this exact decision logic against the training set (first
--train-events records, default 893) with a self-contained simulator that
mirrors testenv/env.py's settle-then-move semantics. Training only ever
touches events[:train_events].

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/heat_blackbox.py" \
      --skip-events 893 --out replay_heat.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate_strategy as strat


class HeatMap:
    def __init__(self, half_life):
        self.decay = 0.5 ** (1.0 / half_life)
        self.heat = {c: 0.0 for c in strat.CELLS}
        self.control = {c: 0.0 for c in strat.CELLS}
        self.home = (1, 1)

    def tick(self, new_positions):
        if self.decay != 1.0:
            h = self.heat
            d = self.decay
            for c in strat.CELLS:
                h[c] *= d
        for p in new_positions:
            self.heat[p] += 1.0
        self.control = {c: sum(self.heat[c2] for c2 in strat.NEIGHBORS[c])
                         for c in strat.CELLS}
        self.home = max(self.control, key=self.control.get)


def decide_move(pos, active, t, heatmap):
    reachable = [b for b in active
                 if strat.manhattan(pos, b['pos']) <= b['expire'] - t]
    if reachable:
        reachable.sort(key=lambda b: (-b['val'], strat.manhattan(pos, b['pos']),
                                       -strat.path_score(pos, b['pos'], heatmap.control)))
        return strat.step_towards(pos, reachable[0]['pos'])
    return strat.greedy_step(pos, heatmap.home, heatmap.control)


def run_sim(events, half_life):
    heatmap = HeatMap(half_life)
    pos = (1, 1)
    active = []
    by_time = {}
    for t, p, v in events:
        by_time.setdefault(t, []).append((p, v))
    t0, t1 = events[0][0], events[-1][0] + strat.BAIT_LIFE
    score = 0.0

    def settle(t):
        nonlocal score, active
        new = by_time.get(t, [])
        for p, v in new:
            active.append({'pos': p, 'val': v, 'expire': t + strat.BAIT_LIFE})
        heatmap.tick([p for p, _ in new])
        remaining = []
        for b in active:
            if b['expire'] < t:
                continue
            if b['pos'] == pos:
                score += b['val']
            else:
                remaining.append(b)
        active = remaining

    for t in range(t0, t1):
        settle(t)
        pos = decide_move(pos, active, t, heatmap)
    settle(t1)
    return score


def train(train_events, half_life_candidates=(30, 60, 120, 300, 600, 1200, 3600, 10**7)):
    best_hl, best_score = None, -1
    for hl in half_life_candidates:
        s = run_sim(train_events, hl)
        print(f'[heat_blackbox] half_life={hl}s  train_score={s:.1f}', file=sys.stderr)
        if s > best_score:
            best_hl, best_score = hl, s
    print(f'[heat_blackbox] chosen half_life={best_hl}s (train_score={best_score:.1f})',
          file=sys.stderr)
    return best_hl


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'))
    ap.add_argument('--train-events', type=int, default=893)
    args = ap.parse_args()

    train_events = strat.load_events(args.data)[:args.train_events]
    half_life = train(train_events)

    # seed the heat map with the full training history (each event ticked
    # in at its own appearance time) so the live phase starts from the
    # same recency-weighted state training ended with.
    heatmap = HeatMap(half_life)
    by_time_train = {}
    for t, p, v in train_events:
        by_time_train.setdefault(t, []).append(p)
    for t in range(train_events[0][0], train_events[-1][0] + 1):
        heatmap.tick(by_time_train.get(t, []))

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
        heatmap.tick(new)
        active = [b for b in active if b['expire'] >= t and b['pos'] != pos]

        nxt = decide_move(pos, active, t, heatmap)
        dx, dy = nxt[0]-pos[0], nxt[1]-pos[1]
        pos = nxt
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
