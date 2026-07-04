"""Black-box strategy: dual-hotspot alternating patrol + greedy ("双热区轮巡").

Idea: the training-set control-power map has two roughly symmetric hotspot
clusters far apart from each other (a single home cell can only cover
~39% of the probability mass and structurally abandons the second
cluster). This strategy computes both peaks -- home1 (global best) and
home2 (the best cell more than 2*RADIUS away from home1, i.e. outside
home1's control range) -- and alternates the idle target between them
every D seconds, greedy-stepping toward whichever is current. Chasing
logic is unchanged: greedily go for the best reachable bait (highest
value; ties by distance, then by undiscounted path control-power).

The only free parameter is the dwell time D (seconds between switches). It
is trained by grid search over a candidate list -- including a
near-infinite D, which degenerates to "always head for home1" as a
built-in single-base baseline -- replaying this exact decision logic
against the training set (first --train-events records, default 893) with
a self-contained simulator mirroring testenv/env.py's settle-then-move
semantics. Training only ever touches events[:train_events].

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/dualbase_blackbox.py" \
      --skip-events 893 --out replay_dualbase.html
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulate_strategy as strat


def find_bases(control):
    home1 = max(control, key=control.get)
    far = [c for c in strat.CELLS if strat.manhattan(c, home1) > 2 * strat.RADIUS]
    home2 = max(far, key=control.get) if far else home1
    return home1, home2


class DualBaseStrategy:
    def __init__(self, home1, home2, D):
        self.home1, self.home2 = home1, home2
        self.D = D
        self.current = home1
        self.switch_at = None   # lazily initialized to the first tick's time

    def idle_step(self, pos, t):
        if self.switch_at is None:
            self.switch_at = t + self.D
        elif t >= self.switch_at:
            self.current = self.home2 if self.current == self.home1 else self.home1
            self.switch_at = t + self.D
        return self.current


def decide_move(pos, active, t, control, dbs):
    reachable = [b for b in active
                 if strat.manhattan(pos, b['pos']) <= b['expire'] - t]
    if reachable:
        reachable.sort(key=lambda b: (-b['val'], strat.manhattan(pos, b['pos']),
                                       -strat.path_score(pos, b['pos'], control)))
        return strat.step_towards(pos, reachable[0]['pos'])
    target = dbs.idle_step(pos, t)
    return strat.greedy_step(pos, target, control)


def run_sim(events, control, home1, home2, D):
    dbs = DualBaseStrategy(home1, home2, D)
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
        pos = decide_move(pos, active, t, control, dbs)
    settle(t1)
    return score


def train(train_events, D_candidates=(30, 60, 120, 300, 600, 900, 1800, 3600, 10**7)):
    counts = {c: 0 for c in strat.CELLS}
    for _, p, _ in train_events:
        counts[p] += 1
    control = strat.control_map(counts, len(train_events))
    home1, home2 = find_bases(control)
    print(f'[dualbase_blackbox] home1={home1} control={control[home1]:.4f}  '
          f'home2={home2} control={control[home2]:.4f}', file=sys.stderr)

    best_D, best_score = None, -1
    for D in D_candidates:
        s = run_sim(train_events, control, home1, home2, D)
        print(f'[dualbase_blackbox] D={D}s  train_score={s:.1f}', file=sys.stderr)
        if s > best_score:
            best_D, best_score = D, s
    print(f'[dualbase_blackbox] chosen D={best_D}s (train_score={best_score:.1f})',
          file=sys.stderr)
    return control, home1, home2, best_D


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'))
    ap.add_argument('--train-events', type=int, default=893)
    args = ap.parse_args()

    train_events = strat.load_events(args.data)[:args.train_events]
    control, home1, home2, D = train(train_events)
    dbs = DualBaseStrategy(home1, home2, D)

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

        nxt = decide_move(pos, active, t, control, dbs)
        dx, dy = nxt[0]-pos[0], nxt[1]-pos[1]
        pos = nxt
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
