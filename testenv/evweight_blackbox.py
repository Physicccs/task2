"""Black-box strategy: expected-value-weighted control field + greedy ("期望值加权控制场").

Idea: simulate_strategy's control map weighs each cell by pure occurrence
frequency. Earlier analysis proved bait value is statistically independent
of position and time in aggregate -- but that is a population-level
result; on a finite 893-event training sample, blending in a little bit of
"how much value has historically shown up near here" could still shift
the estimated best cell towards a slightly richer one, if there's usable
finite-sample structure. This strategy builds two per-cell distributions
from the training set -- occurrence probability and value-share
probability (each cell's fraction of total observed value) -- and blends
them: field(cell) = (1-alpha)*occurrence_prob(cell) + alpha*value_prob(cell),
then control(cell) = sum of field over the usual Manhattan-3 neighborhood,
same as simulate_strategy.control_map. Chasing logic is unchanged
(greedily go for the best reachable bait by actual value); only the
spatial field used for the idle target and the path-score tie-break
changes.

The only free parameter is alpha in [0, 1]. It is trained by grid search
-- alpha=0 exactly reproduces the original occurrence-only model as a
built-in baseline -- replaying this exact decision logic against the
training set (first --train-events records, default 893) with a
self-contained simulator mirroring testenv/env.py's settle-then-move
semantics. Training only ever touches events[:train_events].

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/evweight_blackbox.py" \
      --skip-events 893 --out replay_evweight.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate_strategy as strat


def blended_control(counts, value_sums, total_n, total_v, alpha):
    field = {}
    for c in strat.CELLS:
        occ = counts[c] / total_n if total_n else 0.0
        valp = value_sums[c] / total_v if total_v else 0.0
        field[c] = (1 - alpha) * occ + alpha * valp
    return {c: sum(field[c2] for c2 in strat.NEIGHBORS[c]) for c in strat.CELLS}


def decide_move(pos, active, t, control):
    reachable = [b for b in active
                 if strat.manhattan(pos, b['pos']) <= b['expire'] - t]
    if reachable:
        reachable.sort(key=lambda b: (-b['val'], strat.manhattan(pos, b['pos']),
                                       -strat.path_score(pos, b['pos'], control)))
        return strat.step_towards(pos, reachable[0]['pos'])
    home = max(control, key=control.get)
    return strat.greedy_step(pos, home, control)


def run_sim(events, alpha):
    counts = {c: 0 for c in strat.CELLS}
    value_sums = {c: 0.0 for c in strat.CELLS}
    total_n, total_v = 0, 0.0
    control = blended_control(counts, value_sums, total_n, total_v, alpha)

    pos = (1, 1)
    active = []
    by_time = {}
    for t, p, v in events:
        by_time.setdefault(t, []).append((p, v))
    t0, t1 = events[0][0], events[-1][0] + strat.BAIT_LIFE
    score = 0.0

    def settle(t):
        nonlocal score, active, counts, value_sums, total_n, total_v, control
        new = by_time.get(t, [])
        for p, v in new:
            active.append({'pos': p, 'val': v, 'expire': t + strat.BAIT_LIFE})
            counts[p] += 1
            value_sums[p] += v
            total_n += 1
            total_v += v
        if new:
            control = blended_control(counts, value_sums, total_n, total_v, alpha)
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
        pos = decide_move(pos, active, t, control)
    settle(t1)
    return score


def train(train_events, alpha_candidates=(0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)):
    best_alpha, best_score = None, -1
    for alpha in alpha_candidates:
        s = run_sim(train_events, alpha)
        print(f'[evweight_blackbox] alpha={alpha}  train_score={s:.1f}', file=sys.stderr)
        if s > best_score:
            best_alpha, best_score = alpha, s
    print(f'[evweight_blackbox] chosen alpha={best_alpha} (train_score={best_score:.1f})',
          file=sys.stderr)
    return best_alpha


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'))
    ap.add_argument('--train-events', type=int, default=893)
    args = ap.parse_args()

    train_events = strat.load_events(args.data)[:args.train_events]
    alpha = train(train_events)

    counts = {c: 0 for c in strat.CELLS}
    value_sums = {c: 0.0 for c in strat.CELLS}
    total_n, total_v = 0, 0.0
    for _, p, v in train_events:
        counts[p] += 1
        value_sums[p] += v
        total_n += 1
        total_v += v
    control = blended_control(counts, value_sums, total_n, total_v, alpha)

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
                counts[p] += 1
                value_sums[p] += v
                total_n += 1
                total_v += v
            control = blended_control(counts, value_sums, total_n, total_v, alpha)
        active = [b for b in active if b['expire'] >= t and b['pos'] != pos]

        nxt = decide_move(pos, active, t, control)
        dx, dy = nxt[0]-pos[0], nxt[1]-pos[1]
        pos = nxt
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
