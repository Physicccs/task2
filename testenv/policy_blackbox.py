"""Black-box strategy for testenv/env.py (protocol: see env.py docstring).

Reads INIT / T / bait lines on stdin, answers one "dx dy" per second on
stdout. The decision logic lives in strategy_core.Strategy; the parameters
(waiting cell W, opportunity-cost rate c) come from policy_model.json,
which train_policy.py fits on the first 893 baits only.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy_core import Strategy  # noqa: E402


def main():
    model = json.loads((ROOT / 'policy_model.json').read_text())
    strat = Strategy(W=model['W'], c=model['c'])

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
            new.append((int(x), int(y), float(v)))
        strat.observe(t, new)
        dx, dy = strat.decide(t)
        sys.stdout.write(f'{dx} {dy}\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
