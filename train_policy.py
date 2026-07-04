"""Train the bait-appearance model and tune the strategy on the TRAINING set.

Training data: the first 893 bait records of data.xlsx (the rest is the
held-out test set, consumed only by testenv/env.py --skip-events 893).

Outputs policy_model.json with:
  - prob:   10x10 smoothed appearance-probability map
  - W:      best waiting cell (maximises P(next bait is catchable))
  - c:      opportunity-cost rate (pts/s) tuned by simulating on training data
  - lam,Ev: appearance rate and mean bait value (for the report)

The decision logic here must stay in sync with testenv/policy_blackbox.py
(it is imported by the blackbox at run time via strategy_core).
"""

import json
import re
import sys
from pathlib import Path

import openpyxl

from strategy_core import Strategy, simulate

HERE = Path(__file__).resolve().parent
N_TRAIN = 893


def load_events(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb['Sheet1']
    events = []
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if r[0] is None:
            continue
        m = re.match(r'\((\d+),\s*(\d+)\)', str(r[1]))
        events.append((int(r[0]), int(m.group(1)), int(m.group(2)), float(r[2])))
    events.sort(key=lambda e: e[0])
    return events


def main():
    events = load_events(HERE / 'data.xlsx')
    train = events[:N_TRAIN]
    print(f'training on {len(train)} baits, t = {train[0][0]}..{train[-1][0]} s')

    # --- appearance-probability map (Laplace-smoothed cell frequencies) ---
    counts = {}
    for _, x, y, _ in train:
        counts[(x, y)] = counts.get((x, y), 0) + 1
    n = len(train)
    prob = [[(counts.get((x, y), 0) + 0.5) / (n + 50)
             for y in range(1, 11)] for x in range(1, 11)]

    # --- best waiting cell: maximise P(bait appears within Manhattan dist 3) ---
    def coverage(wx, wy):
        return sum(prob[x - 1][y - 1]
                   for x in range(1, 11) for y in range(1, 11)
                   if abs(x - wx) + abs(y - wy) <= 3)

    W = max(((wx, wy) for wx in range(1, 11) for wy in range(1, 11)),
            key=lambda w: coverage(*w))
    lam = len(train) / (train[-1][0] - train[0][0])
    Ev = sum(v for *_, v in train) / n
    parked = lam * coverage(*W) * Ev
    print(f'W = {W}, coverage = {coverage(*W):.3f}, '
          f'lambda = {lam:.4f}/s, E[v] = {Ev:.2f}, '
          f'parked-rate bound = {60 * parked:.1f} pts/min')

    # --- tune the opportunity-cost rate c on the training set ---
    best = None
    for c in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        strat = Strategy(W=W, c=c)
        score, caught, total, minutes = simulate(train, strat)
        rate = score / minutes
        print(f'  c = {c:.2f}: {rate:7.2f} pts/min, caught {caught}/{total}')
        if best is None or rate > best[0]:
            best = (rate, c)
    rate, c = best
    print(f'chosen c = {c} ({rate:.2f} pts/min on training data)')

    model = {'W': list(W), 'c': c, 'prob': prob,
             'lam': lam, 'Ev': Ev, 'n_train': n,
             'train_pts_per_min': rate}
    (HERE / 'policy_model.json').write_text(json.dumps(model))
    print('model written to policy_model.json')


if __name__ == '__main__':
    sys.exit(main())
