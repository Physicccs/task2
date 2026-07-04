"""Example black-box strategy speaking the test-environment protocol.

Reads bait data on stdin second by second, answers one move (dx, dy) per
second on stdout. Strategy: chase the bait with the highest value among
those still reachable before they expire (ties: nearest first); when
nothing is reachable, drift toward the arena center and wait.

This file is a template — replace decide() with your own strategy.
"""

import sys

GRID_MIN, GRID_MAX = 1, 10
BAIT_LIFE = 3
CENTER = (6, 6)


def clamp(v):
    return max(GRID_MIN, min(GRID_MAX, v))


def step_towards(pos, target):
    """One axis-aligned step from pos toward target (x first)."""
    x, y = pos
    tx, ty = target
    if x != tx:
        return (1 if tx > x else -1, 0)
    if y != ty:
        return (0, 1 if ty > y else -1)
    return (0, 0)


def decide(t, pos, active):
    """active: list of dicts {x, y, v, expire}; return (dx, dy)."""
    reachable = []
    for b in active:
        d = abs(b['x'] - pos[0]) + abs(b['y'] - pos[1])
        if t + d < b['expire']:  # can stand on it while it is still there
            reachable.append((b, d))
    if reachable:
        best, _ = max(reachable, key=lambda bd: (bd[0]['v'], -bd[1]))
        return step_towards(pos, (best['x'], best['y']))
    return step_towards(pos, CENTER)


def main():
    pos = (1, 1)
    active = []
    for line in sys.stdin:
        parts = line.split()
        if not parts or parts[0] == 'INIT':
            continue
        if parts[0] == 'END':
            break
        # "T <t> <k>" followed by k bait lines "x y value"
        t, k = int(parts[1]), int(parts[2])
        for _ in range(k):
            x, y, v = sys.stdin.readline().split()
            active.append({'x': int(x), 'y': int(y), 'v': float(v),
                           'expire': t + BAIT_LIFE})
        # drop expired baits and the one we are standing on (just collected)
        active = [b for b in active
                  if b['expire'] > t and (b['x'], b['y']) != pos]

        dx, dy = decide(t, pos, active)
        pos = (clamp(pos[0] + dx), clamp(pos[1] + dy))
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
