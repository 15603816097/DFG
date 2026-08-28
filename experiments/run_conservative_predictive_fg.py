from __future__ import annotations

import os
import subprocess
import sys


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

STEPS = [
    "experiments/reliability/build_conservative_reliability.py",
    # Compatibility runner deliberately reuses the already-working degraded
    # measurement chain and does not overwrite existing project code.
    "experiments/factor_graph/run_conservative_predictive_fg_compat.py",
    "experiments/evaluation/evaluate_conservative_predictive_fg.py",
]


def run(path):
    full = os.path.join(ROOT, path)
    print()
    print("#" * 112)
    print(path)
    print("#" * 112)
    subprocess.run(
        [sys.executable, full],
        cwd=ROOT,
        check=True,
    )


def main():
    print("=" * 112)
    print("DFG CONSERVATIVE PREDICTIVE FOUR-SENSOR PIPELINE")
    print("=" * 112)

    for step in STEPS:
        run(step)

    print()
    print("=" * 112)
    print("Conservative predictive pipeline finished.")
    print("=" * 112)


if __name__ == "__main__":
    main()
