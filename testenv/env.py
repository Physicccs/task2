"""Black-box test environment for the robot arena competition (Task 2).

The strategy under test is a *black box*: a separate process that talks a
line-based protocol on stdin/stdout. The environment feeds it the test-set
bait data as simulated time advances; after every input the box must answer
with one move (dx, dy).

Protocol (env -> box on the box's stdin, box -> env on its stdout)
------------------------------------------------------------------
  1. INIT line, sent once (no reply expected):
         INIT <n_baits> <t_start> <t_end>
     <n_baits> is the number of bait records in the test set.
  2. One exchange per simulated second t in [t_start, t_end):
         env:  T <t> <k>            (k = number of baits appearing at time t)
               <x> <y> <value>      (k such lines)
         box:  <dx> <dy>
     (dx, dy) must be one of (0,0), (1,0), (-1,0), (0,1), (0,-1).
     Anything else is treated as (0,0) and a warning is printed.
  3. Final line, sent once:
         END

Rules enforced by the environment
---------------------------------
  - The robot starts at START_POS, (1, 1) by default, overridable with
    --start-pos (e.g. to model a robot that gets free preparation time
    before the match to walk to a self-chosen starting cell). Every move
    takes exactly 1 second.
  - Moves that would leave the board are clamped to the boundary [1, 10].
  - A bait appearing at time ta stays for 3 seconds; the robot collects it
    if it stands on the bait's intersection at an integer instant in
    [ta, ta+3] (arriving at the instant of disappearance still counts).

Usage
-----
  python3 testenv/env.py --data data.xlsx \
      --box "python3 testenv/example_blackbox.py" \
      --skip-events 893 --out replay.html --trace trace.json

  --skip-events N   use events[N:] as the test set (default 0 = all events)
  --max-seconds S   truncate the test window to S seconds (default: all)
  --out FILE        write a self-contained HTML replay viewer
  --trace FILE      write the raw trace as JSON
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import openpyxl

GRID_MIN, GRID_MAX = 1, 10
BAIT_LIFE = 3
START_POS = (1, 1)
ALLOWED_MOVES = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}

HERE = Path(__file__).resolve().parent


def load_events(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb['Sheet1']
    events = []
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if r[0] is None:
            continue
        m = re.match(r'\((\d+),\s*(\d+)\)', str(r[1]))
        events.append((int(r[0]), (int(m.group(1)), int(m.group(2))), float(r[2])))
    events.sort(key=lambda e: e[0])
    return events


class BlackBox:
    """A strategy process speaking the line protocol on stdin/stdout."""

    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)

    def send(self, line):
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()

    def recv_move(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('black box closed its stdout before END')
        parts = line.split()
        try:
            move = (int(parts[0]), int(parts[1]))
        except (IndexError, ValueError):
            raise RuntimeError(f'black box sent a malformed move: {line!r}')
        if move not in ALLOWED_MOVES:
            print(f'[env] warning: illegal move {move}, treated as (0,0)',
                  file=sys.stderr)
            move = (0, 0)
        return move

    def close(self):
        try:
            self.send('END')
        except (BrokenPipeError, OSError):
            pass
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def clamp(v):
    return max(GRID_MIN, min(GRID_MAX, v))


def run(events, box, start=START_POS):
    t0 = events[0][0]
    t1 = events[-1][0] + BAIT_LIFE  # last instant worth simulating

    baits = [{'x': p[0], 'y': p[1], 'v': v, 'ta': ta, 'ct': None}
             for ta, p, v in events]
    by_time = {}
    for b in baits:
        by_time.setdefault(b['ta'], []).append(b)

    robot = start
    positions = [robot]          # robot position at instants t0 .. t1
    moves = []                   # accepted move taken during [t, t+1)
    collects = []                # [t, value, cumulative_score]
    score = 0.0
    active = []

    def settle(t):
        """Apply appear/expire/collect for the integer instant t."""
        nonlocal score, active
        active.extend(by_time.get(t, []))
        active = [b for b in active if b['ta'] + BAIT_LIFE >= t]
        remaining = []
        for b in active:
            if (b['x'], b['y']) == robot:
                score += b['v']
                b['ct'] = t
                collects.append([t, b['v'], score])
            else:
                remaining.append(b)
        active = remaining

    box.send(f'INIT {len(baits)} {t0} {t1}')
    for t in range(t0, t1):
        settle(t)
        new = by_time.get(t, [])
        box.send(f'T {t} {len(new)}')
        for b in new:
            box.send(f"{b['x']} {b['y']} {b['v']:g}")
        dx, dy = box.recv_move()
        robot = (clamp(robot[0] + dx), clamp(robot[1] + dy))
        moves.append([dx, dy])
        positions.append(robot)
    settle(t1)
    box.close()

    minutes = (t1 - t0) / 60.0
    return {
        't0': t0, 't1': t1,
        'grid': [GRID_MIN, GRID_MAX],
        'bait_life': BAIT_LIFE,
        'robot': [[x, y] for x, y in positions],
        'moves': moves,
        'baits': baits,
        'collects': collects,
        'score': score,
        'appeared': len(baits),
        'collected': len(collects),
        'minutes': minutes,
        'score_per_min': score / minutes,
        'vmin': min(b['v'] for b in baits),
        'vmax': max(b['v'] for b in baits),
    }


def write_replay_html(trace, out_path):
    template = (HERE / 'viewer_template.html').read_text(encoding='utf-8')
    html = template.replace('__TRACE_JSON__', json.dumps(trace, separators=(',', ':')))
    Path(out_path).write_text(html, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', default='data.xlsx', help='xlsx dataset path')
    ap.add_argument('--box', required=True,
                    help='command line of the black-box strategy process')
    ap.add_argument('--skip-events', type=int, default=0,
                    help='use events[N:] as the test set')
    ap.add_argument('--max-seconds', type=int, default=None,
                    help='truncate the test window to this many seconds')
    ap.add_argument('--start-pos', default=None,
                    help='robot start cell as "x,y" (default: (1,1)); use '
                         'this to model free preparation time before the '
                         'match, i.e. the robot starts already at a '
                         'self-chosen cell instead of the corner')
    ap.add_argument('--trace', default=None, help='write trace JSON here')
    ap.add_argument('--out', default=None, help='write replay HTML here')
    args = ap.parse_args()

    events = load_events(args.data)[args.skip_events:]
    if args.max_seconds is not None:
        cutoff = events[0][0] + args.max_seconds
        events = [e for e in events if e[0] < cutoff]
    if not events:
        sys.exit('empty test set after filtering')

    start = START_POS
    if args.start_pos:
        sx, sy = args.start_pos.split(',')
        start = (int(sx), int(sy))

    print(f'[env] test set: {len(events)} baits, '
          f't = {events[0][0]}..{events[-1][0]} s, start = {start}')
    box = BlackBox(args.box)
    trace = run(events, box, start=start)

    print(f"[env] score = {trace['score']:.1f}  "
          f"score/min = {trace['score_per_min']:.3f}  "
          f"caught {trace['collected']}/{trace['appeared']} "
          f"({trace['collected'] / trace['appeared'] * 100:.1f}%)")

    if args.trace:
        Path(args.trace).write_text(json.dumps(trace), encoding='utf-8')
        print(f'[env] trace written to {args.trace}')
    if args.out:
        write_replay_html(trace, args.out)
        print(f'[env] replay viewer written to {args.out}')


if __name__ == '__main__':
    main()
