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
        "Build LiDAR factor-error dataset",
        "experiments/lidar_predictive/"
        "build_lidar_factor_error_dataset.py",
    ),

    (
        "Train future LiDAR factor-error model",
        "experiments/lidar_predictive/"
        "train_lidar_factor_error_mamba.py",
    ),

    (
        "Evaluate factor-error prediction",
        "experiments/lidar_predictive/"
        "evaluate_lidar_factor_error_prediction.py",
    ),

    (
        "Run factor-error predictive factor graph",
        "experiments/factor_graph/"
        "run_lidar_factor_error_predictive_fg.py",
    ),

    (
        "Evaluate final factor graph",
        "experiments/evaluation/"
        "evaluate_lidar_factor_error_predictive_fg.py",
    ),
]


def main():
    print(
        "=" * 132
    )

    print(
        "DFG LIDAR FACTOR-ERROR-AWARE PREDICTIVE COVARIANCE V1"
    )

    print(
        "=" * 132
    )

    for index, (
        title,
        relative_script,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print()
        print(
            "#" * 132
        )

        print(
            f"[{index}/{len(STEPS)}] "
            f"{title}"
        )

        print(
            "#" * 132
        )

        subprocess.run(
            [
                sys.executable,
                os.path.join(
                    ROOT,
                    relative_script,
                ),
            ],
            cwd=
                ROOT,
            check=
                True,
        )

    print()
    print(
        "=" * 132
    )

    print(
        "LiDAR factor-error predictive V1 pipeline finished."
    )

    print(
        "=" * 132
    )


if __name__ == "__main__":
    main()
