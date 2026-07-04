"""Black-box adapter for the pursue/skip ML classifier trained by solve.py.

Speaks the test-environment protocol of testenv/env.py on stdin/stdout:

  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/solve_blackbox.py" \
      --skip-events 893 --start-pos 5,5 --out replay_solve.html

solve.py labels each bait with an offline-DP-optimal pursue/skip decision,
trains binary classifiers on the first half of the dataset, and saves the
best one (best_model.pkl + scaler.pkl + feature_names.json). This adapter
loads that bundle -- training it first via `python3 solve.py` if the files
are missing -- and replays solve.py's simulate_online() decision loop as a
per-second black box.

Faithfulness to simulate_online(): each bait is evaluated once, when it
appears, from the *committed* robot state (position/time after the last
committed catch, exactly like solve.py); features are built with the same
formulas and fed to the same scaler+model; a positive prediction commits
the robot to the bait and updates the committed state. Between decisions
the robot walks the committed queue one step per second (x before y) and
stays put when idle, so the executed trajectory realizes the committed
schedule exactly.

Catchability rule (the problem statement's, as enforced by env.py): a bait
appearing at ta can be collected at any integer instant in [ta, ta+3]
inclusive, so the robot departing when the bait appears catches anything
up to Manhattan distance 3; while it is still finishing an earlier catch,
the reach is the distance it can cover in the bait's remaining lifetime
after it frees up. The adapter commits to a bait only when this rule says
the catch will land, so every committed catch scores in the environment.

The time_to_next feature (gap to the NEXT bait) cannot be computed at
decision time -- the protocol announces baits only as they appear -- so it
is imputed with the mean appearance gap of the training records
(--train-events, default 893).

Legality guarantees required by the environment: every reply is one of
(0,0)/(1,0)/(-1,0)/(0,1)/(0,-1) as plain ints, positions clamped to
[1, 10]. Coordinates: solve.py's (row, col) parse of "(3,1)" equals the
protocol's (x, y) one-for-one, so no axis remapping is needed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BAIT_LIFE = 3
ALLOWED = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}

sys.path.insert(0, str(ROOT / 'testenv'))
from env import load_events  # noqa: E402


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def load_artifacts():
    paths = [ROOT / 'best_model.pkl', ROOT / 'scaler.pkl',
             ROOT / 'feature_names.json']
    if not all(p.exists() for p in paths):
        print('[solve_blackbox] model artifacts missing, '
              'running solve.py to train (output on stderr)...',
              file=sys.stderr)
        subprocess.run([sys.executable, str(ROOT / 'solve.py')],
                       stdout=sys.stderr, check=True)
    model = joblib.load(paths[0])
    scaler = joblib.load(paths[1])
    feature_names = json.loads(paths[2].read_text(encoding='utf-8'))
    return model, scaler, feature_names


def build_features(bait_time, bait_pos, bait_score, robot_pos, robot_time,
                   seen_t, seen_v, last_bait_time, next_gap, times_max):
    """The exact feature dict of solve.py's simulate_online()."""
    row, col = bait_pos
    dist = manhattan(robot_pos, bait_pos)
    arrival_time = robot_time + dist
    wait_time = max(0, bait_time - arrival_time)
    arrival_at_bait = max(arrival_time, bait_time)
    since_last = bait_time - last_bait_time if last_bait_time is not None else 0

    feat = {
        'dist_to_bait': dist,
        'dist_normalized': min(dist, 18) / 18.0,
        'reachable': 1 if dist <= 3 else 0,
        'can_catch': 1,
        'wait_time': wait_time,
        'arrival_time_delta': min(arrival_at_bait - bait_time, 10),

        'score': bait_score,
        'score_log': np.log1p(bait_score),
        'score_sqrt': np.sqrt(bait_score),
        'score_norm': bait_score / 40.0,
        'is_high_score': 1 if bait_score > 10 else 0,
        'is_vhigh_score': 1 if bait_score > 15 else 0,
        'is_low_score': 1 if bait_score <= 3 else 0,

        'efficiency': bait_score / max(dist, 0.5),

        'row': row,
        'col': col,
        'is_boundary': 1 if (row == 1 or row == 10 or col == 1 or col == 10) else 0,
        'is_corner': 1 if (row in [1, 10] and col in [1, 10]) else 0,
        'dist_to_center': (abs(row - 5.5) + abs(col - 5.5)) / 9.0,

        'robot_row': robot_pos[0] / 10.0,
        'robot_col': robot_pos[1] / 10.0,

        'time_frac': bait_time / max(times_max, 1),
        'time_since_last_bait': since_last,
        'time_since_last_bait_clipped': min(since_last, 15) / 15.0,
        'time_to_next': next_gap,
        'time_to_next_clipped': min(next_gap, 15) / 15.0,

        'past_30s_count': 0.0, 'past_30s_avg': 0.0, 'past_30s_max': 0.0,
        'past_60s_count': 0.0, 'past_60s_avg': 0.0, 'past_60s_max': 0.0,
        'past_120s_count': 0.0, 'past_120s_avg': 0.0, 'past_120s_max': 0.0,

        'opp_score_sum': 0, 'opp_max': 0, 'opp_count': 0,
    }

    # backward windows over baits already announced (t < bait_time)
    for w in (30, 60, 120):
        vals = [v for tt, v in zip(seen_t, seen_v)
                if bait_time - w <= tt < bait_time]
        feat[f'past_{w}s_count'] = float(len(vals))
        if vals:
            feat[f'past_{w}s_avg'] = float(np.mean(vals))
            feat[f'past_{w}s_max'] = float(np.max(vals))
    return feat


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default=str(ROOT / 'data.xlsx'),
                    help='xlsx dataset (training half, for the gap prior)')
    ap.add_argument('--train-events', type=int, default=893,
                    help='leading records used to estimate the mean gap')
    ap.add_argument('--start-pos', default='5,5',
                    help='robot start cell "x,y"; pass the same value to '
                         'testenv/env.py --start-pos (solve.py default 5,5)')
    args = ap.parse_args()

    model, scaler, feature_names = load_artifacts()

    train = load_events(args.data)[:args.train_events]
    gaps = [b[0] - a[0] for a, b in zip(train, train[1:])]
    mean_gap = sum(gaps) / len(gaps) if gaps else 6.0

    sx, sy = args.start_pos.split(',')
    pos = (int(sx), int(sy))            # actual robot cell
    commit_pos, commit_time = pos, 0.0  # state after the last committed catch
    queue = []                          # committed targets [(x, y, expire)]
    seen_t, seen_v = [], []             # announced baits, for window features
    last_bait_time = None
    times_max = 1

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'INIT':
            times_max = int(parts[3]) - BAIT_LIFE  # last appearance time
            continue
        if parts[0] == 'END':
            break

        t, k = int(parts[1]), int(parts[2])
        for _ in range(k):
            x, y, v = sys.stdin.readline().split()
            bpos, bval = (int(x), int(y)), float(v)

            # The bait stays until the instant t+3 inclusive: departing now
            # the robot reaches anything within Manhattan distance 3, and
            # if it is still busy until commit_time, within the lifetime
            # left after that.
            depart = max(commit_time, t)
            dist = manhattan(commit_pos, bpos)
            if depart + dist <= t + BAIT_LIFE:
                feat = build_features(t, bpos, bval, commit_pos, commit_time,
                                      seen_t, seen_v, last_bait_time,
                                      mean_gap, times_max)
                vec = np.array([[feat[c] for c in feature_names]],
                               dtype=np.float64)
                if int(model.predict(scaler.transform(vec))[0]) == 1:
                    queue.append((bpos[0], bpos[1], t + BAIT_LIFE))
                    commit_pos, commit_time = bpos, depart + dist

            seen_t.append(t)
            seen_v.append(bval)
            last_bait_time = t

        # drop targets already reached (env collected them) or expired
        while queue and ((pos[0], pos[1]) == queue[0][:2] or t > queue[0][2]):
            queue.pop(0)

        nxt = pos
        if queue:
            tx, ty, _ = queue[0]
            if pos[0] < tx:
                nxt = (pos[0] + 1, pos[1])
            elif pos[0] > tx:
                nxt = (pos[0] - 1, pos[1])
            elif pos[1] < ty:
                nxt = (pos[0], pos[1] + 1)
            elif pos[1] > ty:
                nxt = (pos[0], pos[1] - 1)

        # legality guard: clamp to the board, force one of the 5 moves
        nxt = (max(1, min(10, nxt[0])), max(1, min(10, nxt[1])))
        dx, dy = nxt[0] - pos[0], nxt[1] - pos[1]
        if (dx, dy) not in ALLOWED:
            dx, dy = 0, 0
            nxt = pos
        pos = nxt

        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
