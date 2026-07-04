"""Black-box strategy: distance-normalized value-rate gate + greedy ("距离价值密度过滤").

Idea: an earlier experiment showed a flat value threshold (ignore every
bait worth less than T) always hurts, monotonically, because chasing any
reachable bait is essentially free -- travel cost doesn't scale with how
far away the *next* opportunity after that is. But the threshold in that
experiment ignored distance entirely, so it penalized a "1-point bait
right next to me" exactly as harshly as a "1-point bait 3 cells away".
This strategy gates on value *density* instead: rate = value / max(distance, 1).
A bait is only worth actively detouring for if rate >= rho; baits below
the gate are simply not targeted (but are still picked up for free if the
robot's path happens to cross them anyway -- matching the earlier
"仅在顺路时捡" rule). This should behave very differently from the flat
threshold: cheap-to-reach low-value baits stay attractive (rate is high
when distance is small), while it becomes more selective specifically
about spending several seconds of travel on low-value targets.

The only free parameter is rho. It is trained by grid search -- rho=0
exactly reproduces the original "chase anything reachable" baseline --
replaying this exact decision logic against the training set (first
--train-events records, default 893) with a self-contained simulator
mirroring testenv/env.py's settle-then-move semantics. Training only ever
touches events[:train_events].

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/ratefilter_blackbox.py" \
      --skip-events 893 --out replay_ratefilter.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate_strategy as strat


def decide_move(pos, active, t, control, rho):
    reachable = [b for b in active
                 if strat.manhattan(pos, b['pos']) <= b['expire'] - t]
    gated = [b for b in reachable
             if b['val'] / max(strat.manhattan(pos, b['pos']), 1) >= rho]
    if gated:
        gated.sort(key=lambda b: (-b['val'], strat.manhattan(pos, b['pos']),
                                   -strat.path_score(pos, b['pos'], control)))
        return strat.step_towards(pos, gated[0]['pos'])
    home = max(control, key=control.get)
    return strat.greedy_step(pos, home, control)


def run_sim(events, rho):
    counts = {c: 0 for c in strat.CELLS}
    total = 0
    control = strat.control_map(counts, total)

    pos = (1, 1)
    active = []
    by_time = {}
    for t, p, v in events:
        by_time.setdefault(t, []).append((p, v))
    t0, t1 = events[0][0], events[-1][0] + strat.BAIT_LIFE
    score = 0.0

    def settle(t):
        nonlocal score, active, counts, total, control
        new = by_time.get(t, [])
        for p, v in new:
            active.append({'pos': p, 'val': v, 'expire': t + strat.BAIT_LIFE})
            counts[p] += 1
            total += 1
        if new:
            control = strat.control_map(counts, total)
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
        pos = decide_move(pos, active, t, control, rho)
    settle(t1)
    return score


def train(train_events, rho_candidates=(0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)):
    best_rho, best_score = None, -1
    for rho in rho_candidates:
        s = run_sim(train_events, rho)
        print(f'[ratefilter_blackbox] rho={rho}  train_score={s:.1f}', file=sys.stderr)
        if s > best_score:
            best_rho, best_score = rho, s
    print(f'[ratefilter_blackbox] chosen rho={best_rho} (train_score={best_score:.1f})',
          file=sys.stderr)
    return best_rho


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'))
    ap.add_argument('--train-events', type=int, default=893)
    args = ap.parse_args()

    train_events = strat.load_events(args.data)[:args.train_events]
    rho = train(train_events)

    counts = {c: 0 for c in strat.CELLS}
    for _, p, _ in train_events:
        counts[p] += 1
    total = len(train_events)
    control = strat.control_map(counts, total)

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

        nxt = decide_move(pos, active, t, control, rho)
        dx, dy = nxt[0]-pos[0], nxt[1]-pos[1]
        pos = nxt
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
