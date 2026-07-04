"""Core decision logic of the robot strategy + a faithful local simulator.

Shared by train_policy.py (tuning on the training set) and
testenv/policy_blackbox.py (the black box run by testenv/env.py), so the
tuned behaviour and the tested behaviour are literally the same code.

Strategy ("hotspot-anchored greedy with opportunity cost"):
  - Track the baits currently on the board. A bait appearing at ta can be
    collected by standing on it at any integer instant in [ta, ta+3].
  - Consider every single bait and every ordered pair of baits that the
    robot can still reach in time. Plan utility =
        sum of values  -  c * (extra seconds spent away from W)
    where W is the waiting cell that maximises the probability that the
    next bait appears within catching range, and c (pts/s) is the expected
    scoring rate given up while off-post (tuned on training data).
  - Follow the best positive-utility plan; otherwise return to W and wait.
"""

from itertools import permutations

GRID_MIN, GRID_MAX = 1, 10
BAIT_LIFE = 3
START_POS = (1, 1)


def clamp(v):
    return max(GRID_MIN, min(GRID_MAX, v))


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Strategy:
    def __init__(self, W=(3, 8), c=0.4):
        self.W = tuple(W)
        self.c = c
        self.pos = START_POS
        self.active = []  # dicts {x, y, v, ta}

    def observe(self, t, new_baits):
        """new_baits: iterable of (x, y, v) appearing at time t."""
        for x, y, v in new_baits:
            self.active.append({'x': x, 'y': y, 'v': v, 'ta': t})
        # drop what is gone by the time we could arrive (>= t+1), and what
        # we are standing on (the environment has just collected it)
        self.active = [b for b in self.active
                       if b['ta'] + BAIT_LIFE >= t + 1
                       and (b['x'], b['y']) != self.pos]

    def _plan_utility(self, t, order):
        """Total value and end position of visiting baits in this order;
        None if some bait expires before we can stand on it."""
        pos, now, value = self.pos, t, 0.0
        for b in order:
            now += dist(pos, (b['x'], b['y']))
            if now > b['ta'] + BAIT_LIFE:
                return None
            pos = (b['x'], b['y'])
            value += b['v']
        extra = (now - t) + dist(pos, self.W) - dist(self.pos, self.W)
        return value - self.c * max(0, extra), (b['x'], b['y'])

    def decide(self, t):
        """Return (dx, dy) for the second [t, t+1)."""
        best, target = 0.0, self.W
        candidates = self.active[-6:]  # board never holds many live baits
        plans = [(b,) for b in candidates]
        plans += list(permutations(candidates, 2))
        for order in plans:
            res = self._plan_utility(t, order)
            if res and res[0] > best:
                best, target = res[0], (order[0]['x'], order[0]['y'])
        return self._step(target)

    def _step(self, target):
        """One axis-aligned step toward target; among equally good steps
        prefer the one that also moves toward the waiting cell W."""
        options = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)):
            np = (clamp(self.pos[0] + dx), clamp(self.pos[1] + dy))
            options.append((dist(np, target), dist(np, self.W), (dx, dy), np))
        _, _, move, np = min(options)
        self.pos = np
        return move


def simulate(events, strat):
    """Replay `events` [(t, x, y, v), ...] under the exact rules of
    testenv/env.py. Returns (score, caught, total, minutes)."""
    by_time = {}
    for ta, x, y, v in events:
        by_time.setdefault(ta, []).append({'x': x, 'y': y, 'v': v, 'ta': ta})
    t0, t1 = events[0][0], events[-1][0] + BAIT_LIFE

    robot = START_POS
    score, caught, active = 0.0, 0, []

    def settle(t):
        nonlocal score, caught, active
        active.extend(by_time.get(t, []))
        active = [b for b in active if b['ta'] + BAIT_LIFE >= t]
        keep = []
        for b in active:
            if (b['x'], b['y']) == robot:
                score += b['v']
                caught += 1
            else:
                keep.append(b)
        active = keep

    for t in range(t0, t1):
        settle(t)
        strat.observe(t, [(b['x'], b['y'], b['v']) for b in by_time.get(t, [])])
        dx, dy = strat.decide(t)
        robot = (clamp(robot[0] + dx), clamp(robot[1] + dy))
        assert robot == strat.pos
    settle(t1)

    minutes = (t1 - t0) / 60.0
    return score, caught, len(events), minutes
