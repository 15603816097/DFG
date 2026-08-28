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
        "Run axis-aware covariance ablation",
        os.path.join(
            ROOT,
            "experiments",
            "factor_graph",
            "run_axis_aware_covariance_ablation.py",
        ),
    ),
    (
        "Evaluate axis-aware covariance",
        os.path.join(
            ROOT,
            "experiments",
            "evaluation",
            "evaluate_axis_aware_covariance.py",
        ),
    ),
]


def main():
    print(
        "=" * 120
    )

    print(
        "DFG AXIS-AWARE / DOF-SPECIFIC "
        "DYNAMIC COVARIANCE PIPELINE"
    )

    print(
        "=" * 120
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
            "#" * 120
        )

        print(
            f"[{index}/{len(STEPS)}] "
            f"{title}"
        )

        print(
            "#" * 120
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
        "Axis-aware covariance pipeline finished."
    )


if __name__ == "__main__":
    main()
