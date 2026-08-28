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
        "Build LiDAR event trigger",
        os.path.join(
            ROOT,
            "experiments",
            "reliability",
            "build_lidar_event_triggered_reliability.py",
        ),
    ),

    (
        "Run LiDAR event-triggered factor graph",
        os.path.join(
            ROOT,
            "experiments",
            "factor_graph",
            "run_lidar_event_triggered_fg.py",
        ),
    ),

    (
        "Evaluate LiDAR event-triggered factor graph",
        os.path.join(
            ROOT,
            "experiments",
            "evaluation",
            "evaluate_lidar_event_triggered_fg.py",
        ),
    ),
]


def main():
    print(
        "=" * 124
    )

    print(
        "DFG LIDAR EVENT-TRIGGERED RELIABILITY PIPELINE"
    )

    print(
        "=" * 124
    )

    for index, (
        title,
        script,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print()

        print(
            "#" * 124
        )

        print(
            f"[{index}/{len(STEPS)}] "
            f"{title}"
        )

        print(
            "#" * 124
        )

        subprocess.run(
            [
                sys.executable,
                script,
            ],
            cwd=ROOT,
            check=True,
        )

    print()

    print(
        "LiDAR event-triggered pipeline finished."
    )


if __name__ == "__main__":
    main()
