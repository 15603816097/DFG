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
    (
        "Build cross-sensor coordinated reliability",
        os.path.join(
            ROOT,
            "experiments",
            "reliability",
            "build_cross_sensor_coordinated_reliability_v1.py",
        ),
    ),
    (
        "Run coordinated four-sensor factor graph",
        os.path.join(
            ROOT,
            "experiments",
            "factor_graph",
            "run_cross_sensor_coordinated_fg_v1.py",
        ),
    ),
    (
        "Evaluate coordinated factor graph",
        os.path.join(
            ROOT,
            "experiments",
            "evaluation",
            "evaluate_cross_sensor_coordination_v1.py",
        ),
    ),
]


def main():
    print("=" * 120)
    print("DFG SENSOR-SPECIFIC + CROSS-SENSOR COORDINATION V1 PIPELINE")
    print("=" * 120)

    for i, (title, script) in enumerate(STEPS, start=1):
        print()
        print("#" * 120)
        print(f"[{i}/{len(STEPS)}] {title}")
        print("#" * 120)

        subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            check=True,
        )

    print()
    print("=" * 120)
    print("Cross-sensor coordination V1 pipeline finished.")
    print("=" * 120)


if __name__ == "__main__":
    main()
