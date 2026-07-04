"""Black-box strategy: perimeter-patrol + greedy chase ("环绕型+贪心").

Idea: instead of freezing at a single "home" cell when idle, walk a fixed
rectangular patrol loop centered on the training-set hotspot, so the robot
keeps sweeping past nearby high-probability cells rather than sitting
still. Whenever a bait is reachable, greedily chase the best one (highest
value; ties broken by distance, then by undiscounted path control-power),
exactly as in strategy_blackbox.py; when nothing is reachable, take the
next step along the patrol loop (rejoining it at the nearest point if a
chase left the robot off-loop).

The patrol half-width R is the only free parameter. It is trained by grid
search over R = 1..6, replaying this exact decision logic against the
training set (first --train-events records, default 893) with a
self-contained simulator that mirrors testenv/env.py's settle-then-move
semantics, and keeping the R with the highest training score. Training
only ever touches events[:train_events]; the held-out test set is never
read until the live protocol loop runs.

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/patrol_blackbox.py" \
      --skip-events 893 --out replay_patrol.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate_strategy as strat


def build_loop(home, R):
    """Axis-aligned rectangle boundary centered on home, half-width R,
    clipped to the 1..10 grid, walkable as a single step-by-step cycle."""
    x0, x1 = max(1, home[0]-R), min(10, home[0]+R)
    y0, y1 = max(1, home[1]-R), min(10, home[1]+R)
    if x0 == x1 or y0 == y1:
        return [home]
    loop = []
    for x in range(x0, x1):          # bottom edge, left -> right
        loop.append((x, y0))
    for y in range(y0, y1):          # right edge, bottom -> top
        loop.append((x1, y))
    for x in range(x1, x0, -1):      # top edge, right -> left
        loop.append((x, y1))
    for y in range(y1, y0, -1):      # left edge, top -> bottom
        loop.append((x0, y))
    return loop


class PatrolStrategy:
    def __init__(self, home, R):
        self.loop = build_loop(home, R)
        self.idx = min(range(len(self.loop)),
                        key=lambda i: strat.manhattan(home, self.loop[i]))

    def idle_step(self, pos):
        target = self.loop[self.idx]
        if pos == target:
            self.idx = (self.idx + 1) % len(self.loop)
            target = self.loop[self.idx]
        if pos == target:
            return pos
        return strat.step_towards(pos, target)


def decide_move(pos, active, t, control, idle_strategy):
    """Shared decision logic used identically by the trainer and the live
    protocol loop: chase the best reachable bait, else patrol."""
    reachable = [b for b in active
                 if strat.manhattan(pos, b['pos']) <= b['expire'] - t]
    if reachable:
        reachable.sort(key=lambda b: (-b['val'], strat.manhattan(pos, b['pos']),
                                       -strat.path_score(pos, b['pos'], control)))
        return strat.step_towards(pos, reachable[0]['pos'])
    return idle_strategy.idle_step(pos)


def run_sim(events, home, control, R):
    """Self-contained replay mirroring testenv/env.py's settle-then-move
    semantics, used only to score a candidate R on the training set."""
    ps = PatrolStrategy(home, R)
    pos = (1, 1)
    active = []
    by_time = {}
    for t, p, v in events:
        by_time.setdefault(t, []).append((p, v))
    t0, t1 = events[0][0], events[-1][0] + strat.BAIT_LIFE
    score = 0.0

    def settle(t):
        nonlocal score, active
        for p, v in by_time.get(t, []):
            active.append({'pos': p, 'val': v, 'expire': t + strat.BAIT_LIFE})
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
        pos = decide_move(pos, active, t, control, ps)
    settle(t1)
    return score


def train(train_events, R_candidates=range(1, 7)):
    counts = {c: 0 for c in strat.CELLS}
    for _, p, _ in train_events:
        counts[p] += 1
    control = strat.control_map(counts, len(train_events))
    home = max(control, key=control.get)

    best_R, best_score = None, -1
    for R in R_candidates:
        s = run_sim(train_events, home, control, R)
        print(f'[patrol_blackbox] R={R}  train_score={s:.1f}', file=sys.stderr)
        if s > best_score:
            best_R, best_score = R, s
    print(f'[patrol_blackbox] chosen R={best_R} (train_score={best_score:.1f})',
          file=sys.stderr)
    return home, control, best_R


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'))
    ap.add_argument('--train-events', type=int, default=893)
    args = ap.parse_args()

    train_events = strat.load_events(args.data)[:args.train_events]
    home, control, R = train(train_events)
    ps = PatrolStrategy(home, R)

    pos = (1, 1)
    active = []
    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == 'INIT':
            continue
        if parts[0] == 'END':
            break
        t, k = int(parts[1]), int(parts[2])
        for _ in range(k):
            x, y, v = sys.stdin.readline().split()
            active.append({'pos': (int(x), int(y)), 'val': float(v),
                            'expire': t + strat.BAIT_LIFE})
        active = [b for b in active if b['expire'] >= t and b['pos'] != pos]

        nxt = decide_move(pos, active, t, control, ps)
        dx, dy = nxt[0]-pos[0], nxt[1]-pos[1]
        pos = nxt
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
