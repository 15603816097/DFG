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
        "Predictive sensor-wise ablation",
        "experiments/factor_graph/"
        "run_sensorwise_predictive_ablation.py",
    ),

    (
        "Oracle sensor-wise ablation",
        "experiments/factor_graph/"
        "run_sensorwise_oracle_ablation.py",
    ),

    (
        "Evaluation",
        "experiments/evaluation/"
        "evaluate_sensorwise_reliability_ablation.py",
    ),
]


def main():
    print("=" * 120)
    print(
        "DFG SENSOR-WISE RELIABILITY ABLATION PIPELINE"
    )
    print("=" * 120)

    for index, (
        title,
        relative_path,
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
                os.path.join(
                    ROOT,
                    relative_path,
                ),
            ],
            cwd=ROOT,
            check=True,
        )

    print()
    print(
        "Sensor-wise reliability ablation finished."
    )


if __name__ == "__main__":
    main()
