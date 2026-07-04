import re
import openpyxl
import numpy as np

GRID = 10
RADIUS = 3          # control range: Manhattan distance <= 3
BAIT_LIFE = 3        # seconds a bait stays visible
SPEED = 1            # m/s == 1 grid step / second

def load_events(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb['Sheet1']
    rows = list(ws.iter_rows(values_only=True))[1:]
    events = []
    for r in rows:
        if r[0] is None:
            continue
        t, pos, val = r[0], r[1], r[2]
        m = re.match(r'\((\d+),\s*(\d+)\)', str(pos))
        x, y = int(m.group(1)), int(m.group(2))
        events.append((int(t), (x, y), float(val)))
    events.sort(key=lambda e: e[0])
    return events

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

CELLS = [(x, y) for x in range(1, GRID+1) for y in range(1, GRID+1)]
NEIGHBORS = {c: [c2 for c2 in CELLS if manhattan(c, c2) <= RADIUS] for c in CELLS}

def control_map(counts, total):
    if total == 0:
        prob = {c: 1.0/len(CELLS) for c in CELLS}
    else:
        prob = {c: counts[c]/total for c in CELLS}
    return {c: sum(prob[c2] for c2 in NEIGHBORS[c]) for c in CELLS}

def greedy_step(start, goal, control):
    """Idle/wander move: among the (<=2) neighbor cells that make monotone
    progress toward goal, step to whichever has higher control power."""
    if start == goal:
        return start
    dx, dy = goal[0]-start[0], goal[1]-start[1]
    candidates = []
    if dx != 0:
        candidates.append((start[0] + (1 if dx > 0 else -1), start[1]))
    if dy != 0:
        candidates.append((start[0], start[1] + (1 if dy > 0 else -1)))
    return max(candidates, key=lambda c: control[c])

def path_score(a, b, control):
    """best achievable (direction-invariant) sum of control power over any
    monotone shortest path from a to b; used only as a tie-breaker."""
    D = manhattan(a, b)
    if D == 0:
        return control[a]
    dx, dy = b[0]-a[0], b[1]-a[1]
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    nx, ny = abs(dx), abs(dy)
    g = np.zeros((nx+1, ny+1))
    def cell(i, j):
        return (a[0]+i*sx, a[1]+j*sy)
    for i in range(nx, -1, -1):
        for j in range(ny, -1, -1):
            v = control[cell(i, j)]
            best = -np.inf
            if i < nx:
                best = max(best, g[i+1, j])
            if j < ny:
                best = max(best, g[i, j+1])
            g[i, j] = v + (0 if best == -np.inf else best)
    return g[0, 0]

def step_towards(cur, target):
    x, y = cur
    tx, ty = target
    if x != tx:
        x += 1 if tx > x else -1
    elif y != ty:
        y += 1 if ty > y else -1
    return (x, y)

def simulate(events, T=0, verbose=False):
    T_end = max(e[0] for e in events) + BAIT_LIFE + 1
    by_time = {}
    for t, pos, val in events:
        by_time.setdefault(t, []).append((pos, val))

    counts = {c: 0 for c in CELLS}
    total = 0
    control = control_map(counts, total)
    home = max(control, key=control.get)

    robot = home
    active = []          # list of dicts: pos, val, expire (exclusive) -- ALL baits, incl. below T
    score = 0.0
    collected = 0
    appeared = 0

    for t in range(T_end):
        if t in by_time:
            for pos, val in by_time[t]:
                appeared += 1
                active.append({'pos': pos, 'val': val, 'expire': t + BAIT_LIFE})
                if val >= T:
                    counts[pos] += 1
                    total += 1
            control = control_map(counts, total)
            home = max(control, key=control.get)

        # drop expired baits
        active = [b for b in active if b['expire'] > t]

        # reachable = can still get there before it expires, AND worth actively chasing (val>=T)
        reachable = [b for b in active
                     if b['val'] >= T and manhattan(robot, b['pos']) <= (b['expire'] - t)]

        if reachable:
            reachable.sort(key=lambda b: (-b['val'],
                                           manhattan(robot, b['pos']),
                                           -path_score(robot, b['pos'], control)))
            target = reachable[0]['pos']
            robot = step_towards(robot, target)
        else:
            robot = greedy_step(robot, home, control)

        # collect anything sitting on the robot's new cell
        still_active = []
        for b in active:
            if b['pos'] == robot and b['expire'] > t:
                score += b['val']
                collected += 1
            else:
                still_active.append(b)
        active = still_active

    minutes = T_end / 60.0
    return {
        'T': T, 'score': score, 'minutes': minutes,
        'score_per_min': score/minutes, 'appeared': appeared,
        'collected': collected, 'catch_rate': collected/appeared,
    }

def simulate_train_test(events, n_train, T=0):
    """Build the initial probability model from the first n_train events
    (by appearance time), then run the strategy live on the remaining
    events, still updating the model online as the test events occur."""
    train, test = events[:n_train], events[n_train:]

    counts = {c: 0 for c in CELLS}
    for _, pos, val in train:
        if val >= T:
            counts[pos] += 1
    total = sum(counts.values())
    control = control_map(counts, total)
    home = max(control, key=control.get)

    t0 = test[0][0]
    T_end = max(e[0] for e in test) + BAIT_LIFE + 1

    by_time = {}
    for t, pos, val in test:
        by_time.setdefault(t, []).append((pos, val))

    robot = home
    active = []
    score = 0.0
    collected = 0
    appeared = 0

    for t in range(t0, T_end):
        if t in by_time:
            for pos, val in by_time[t]:
                appeared += 1
                active.append({'pos': pos, 'val': val, 'expire': t + BAIT_LIFE})
                if val >= T:
                    counts[pos] += 1
                    total += 1
            control = control_map(counts, total)
            home = max(control, key=control.get)

        active = [b for b in active if b['expire'] > t]
        reachable = [b for b in active
                     if b['val'] >= T and manhattan(robot, b['pos']) <= (b['expire'] - t)]

        if reachable:
            reachable.sort(key=lambda b: (-b['val'],
                                           manhattan(robot, b['pos']),
                                           -path_score(robot, b['pos'], control)))
            target = reachable[0]['pos']
            robot = step_towards(robot, target)
        else:
            robot = greedy_step(robot, home, control)

        still_active = []
        for b in active:
            if b['pos'] == robot and b['expire'] > t:
                score += b['val']
                collected += 1
            else:
                still_active.append(b)
        active = still_active

    minutes = (T_end - t0) / 60.0
    return {
        'T': T, 'n_train': n_train, 'n_test': len(test),
        'train_home': home_from_counts(train, T),
        'score': score, 'minutes': minutes,
        'score_per_min': score/minutes, 'appeared': appeared,
        'collected': collected, 'catch_rate': collected/appeared,
        'test_total_value': sum(v for _, _, v in test),
    }

def home_from_counts(train, T=0):
    counts = {c: 0 for c in CELLS}
    for _, pos, val in train:
        if val >= T:
            counts[pos] += 1
    control = control_map(counts, sum(counts.values()))
    return max(control, key=control.get)

if __name__ == '__main__':
    events = load_events('data.xlsx')
    print(f"loaded {len(events)} events, span {max(e[0] for e in events)/60:.1f} min")

    print("\n--- full-dataset online run (no train/test split) ---")
    r = simulate(events)
    print(f"score={r['score']:.1f}  score/min={r['score_per_min']:.3f}  "
          f"catch_rate={r['catch_rate']*100:.1f}%  ({r['collected']}/{r['appeared']})")

    print("\n--- train(first 893) / test(remaining) split ---")
    r2 = simulate_train_test(events, 893)
    print(f"train_home={r2['train_home']}  score={r2['score']:.1f}  score/min={r2['score_per_min']:.3f}  "
          f"catch_rate={r2['catch_rate']*100:.1f}%  ({r2['collected']}/{r2['appeared']})  "
          f"test_max={r2['test_total_value']:.0f}")

    print("\n--- training T (ignore-below-threshold) on the training set ---")
    train_events = events[:893]
    t_results = []
    for T in range(0, 41):
        r = simulate(train_events, T=T)
        t_results.append(r)
        print(f"T={T:2d}  train_score={r['score']:7.1f}  score/min={r['score_per_min']:6.3f}  "
              f"catch_rate={r['catch_rate']*100:5.1f}%  ({r['collected']}/{r['appeared']})")
    best_t = max(t_results, key=lambda r: r['score'])
    print(f"\nbest T = {best_t['T']}  train_score={best_t['score']:.1f}  "
          f"train_score/min={best_t['score_per_min']:.3f}")

    print(f"\n--- applying T={best_t['T']} to the held-out test set ---")
    r_test = simulate_train_test(events, 893, T=best_t['T'])
    print(f"T={r_test['T']}  test_score={r_test['score']:7.1f}  score/min={r_test['score_per_min']:6.3f}  "
          f"catch_rate={r_test['catch_rate']*100:5.1f}%  ({r_test['collected']}/{r_test['appeared']})")
