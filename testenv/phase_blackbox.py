"""Black-box strategy: periodicity-phase-aware patrol + greedy ("周期相位感知型").

Idea: inter-arrival times were shown (cwt_analysis.py) to be a bounded,
quasi-periodic process with a cycle around a few seconds, not a memoryless
Poisson process -- appearances cluster more predictably in time than a
pure-spatial model can use. This strategy tracks the elapsed time since
the last bait appeared and switches idle behavior on that signal: right
after an appearance (elapsed < tau) a new one is not "due" yet by the
quasi-periodic pattern, so the robot can afford to drift toward the
secondary hotspot (home2, the best cell outside home1's control range) to
opportunistically cover it; once elapsed >= tau a new appearance is
statistically due soon, so the robot tightens back onto the single best
cell (home1) to maximize the chance of being in range when it fires.
Chasing logic is unchanged: greedily go for the best reachable bait.

The only free parameter is tau (seconds). It is trained by grid search
over a candidate list -- including tau=0, which degenerates to "always at
home1" -- replaying this exact decision logic against the training set
(first --train-events records, default 893) with a self-contained
simulator mirroring testenv/env.py's settle-then-move semantics. Training
only ever touches events[:train_events].

Protocol: see testenv/env.py.
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/phase_blackbox.py" \
      --skip-events 893 --out replay_phase.html
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


class PhaseStrategy:
    def __init__(self, home1, home2, tau):
        self.home1, self.home2, self.tau = home1, home2, tau
        self.last_appear = None

    def on_appear(self, t):
        self.last_appear = t

    def idle_step(self, pos, t, control):
        if self.last_appear is None:
            target = self.home1
        else:
            elapsed = t - self.last_appear
            target = self.home2 if elapsed < self.tau else self.home1
        return strat.greedy_step(pos, target, control)


def decide_move(pos, active, t, control, ps):
    reachable = [b for b in active
                 if strat.manhattan(pos, b['pos']) <= b['expire'] - t]
    if reachable:
        reachable.sort(key=lambda b: (-b['val'], strat.manhattan(pos, b['pos']),
                                       -strat.path_score(pos, b['pos'], control)))
        return strat.step_towards(pos, reachable[0]['pos'])
    return ps.idle_step(pos, t, control)


def run_sim(events, control, home1, home2, tau):
    ps = PhaseStrategy(home1, home2, tau)
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
        if new:
            ps.on_appear(t)
        for p, v in new:
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


def train(train_events, tau_candidates=(0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30)):
    counts = {c: 0 for c in strat.CELLS}
    for _, p, _ in train_events:
        counts[p] += 1
    control = strat.control_map(counts, len(train_events))
    home1, home2 = find_bases(control)
    print(f'[phase_blackbox] home1={home1}  home2={home2}', file=sys.stderr)

    best_tau, best_score = None, -1
    for tau in tau_candidates:
        s = run_sim(train_events, control, home1, home2, tau)
        print(f'[phase_blackbox] tau={tau}s  train_score={s:.1f}', file=sys.stderr)
        if s > best_score:
            best_tau, best_score = tau, s
    print(f'[phase_blackbox] chosen tau={best_tau}s (train_score={best_score:.1f})',
          file=sys.stderr)
    return control, home1, home2, best_tau


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(root / 'data.xlsx'))
    ap.add_argument('--train-events', type=int, default=893)
    args = ap.parse_args()

    train_events = strat.load_events(args.data)[:args.train_events]
    control, home1, home2, tau = train(train_events)
    ps = PhaseStrategy(home1, home2, tau)

    pos = (1, 1)
    active = []
    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == 'INIT':
            continue
        if parts[0] == 'END':
            break
        t, k = int(parts[1]), int(parts[2])
        if k:
            ps.on_appear(t)
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
