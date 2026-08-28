from __future__ import annotations

import os
from pathlib import Path

import numpy as np


ROOT = Path(
    __file__
).resolve().parents[
    2
]

GROUND_TRUTH = (
    ROOT
    /
    "results"
    /
    "ground_truth"
    /
    "trajectory.txt"
)

METHODS = [
    (
        "Physical LiDAR Fixed",
        ROOT
        /
        "results"
        /
        "lidar_physical_oracle_fg"
        /
        "physical_lidar_fixed"
        /
        "trajectory.txt",
    ),

    (
        "Factor-Error Predictive LiDAR",
        ROOT
        /
        "results"
        /
        "lidar_factor_error_predictive_fg_v1"
        /
        "trajectory.txt",
    ),

    (
        "Physical LiDAR GT-Oracle Dynamic",
        ROOT
        /
        "results"
        /
        "lidar_physical_oracle_fg"
        /
        "physical_lidar_oracle"
        /
        "trajectory.txt",
    ),
]


def metrics(
    gt,
    trajectory,
):
    n = min(
        len(
            gt
        ),
        len(
            trajectory
        ),
    )

    error = (
        trajectory[
            :n,
            :3
        ]
        -
        gt[
            :n,
            :3
        ]
    )

    e3 = np.linalg.norm(
        error,
        axis=1,
    )

    e2 = np.linalg.norm(
        error[
            :,
            :2
        ],
        axis=1,
    )

    return {
        "ATE3D":
            float(
                np.sqrt(
                    np.mean(
                        e3
                        **
                        2
                    )
                )
            ),

        "ATE2D":
            float(
                np.sqrt(
                    np.mean(
                        e2
                        **
                        2
                    )
                )
            ),

        "Mean3D":
            float(
                np.mean(
                    e3
                )
            ),

        "Max3D":
            float(
                np.max(
                    e3
                )
            ),

        "ZRMSE":
            float(
                np.sqrt(
                    np.mean(
                        error[
                            :,
                            2
                        ]
                        **
                        2
                    )
                )
            ),
    }


def main():
    gt = np.loadtxt(
        GROUND_TRUTH,
        dtype=np.float64,
    )

    print(
        "=" * 128
    )

    print(
        "LIDAR FACTOR-ERROR-AWARE PREDICTIVE FG EVALUATION"
    )

    print(
        "=" * 128
    )

    results = {}

    for name, path in METHODS:
        if not path.exists():
            print(
                f"{name:40s}: NOT RUN"
            )

            continue

        result = metrics(
            gt,
            np.loadtxt(
                path,
                dtype=np.float64,
            ),
        )

        results[
            name
        ] = result

        print(
            f"{name:40s} "
            f"ATE3D={result['ATE3D']:.6f} "
            f"ATE2D={result['ATE2D']:.6f} "
            f"Mean3D={result['Mean3D']:.6f} "
            f"Max3D={result['Max3D']:.6f} "
            f"ZRMSE={result['ZRMSE']:.6f}"
        )

    fixed = results.get(
        "Physical LiDAR Fixed"
    )

    pred = results.get(
        "Factor-Error Predictive LiDAR"
    )

    oracle = results.get(
        "Physical LiDAR GT-Oracle Dynamic"
    )

    print()

    if fixed and pred:
        improvement = (
            fixed[
                "ATE3D"
            ]
            -
            pred[
                "ATE3D"
            ]
        ) / fixed[
            "ATE3D"
        ] * 100.0

        print(
            "Predictive improvement vs Physical Fixed:",
            f"{improvement:+.2f}%",
        )

    if pred and oracle:
        gap = (
            pred[
                "ATE3D"
            ]
            -
            oracle[
                "ATE3D"
            ]
        ) / oracle[
            "ATE3D"
        ] * 100.0

        print(
            "Predictive gap from GT Oracle:",
            f"{gap:+.2f}%",
        )

    output = (
        ROOT
        /
        "results"
        /
        "lidar_factor_error_predictive_fg_evaluation.txt"
    )

    output.write_text(
        "\n".join(
            [
                f"{name},"
                f"{result['ATE3D']:.10f},"
                f"{result['ATE2D']:.10f},"
                f"{result['Mean3D']:.10f},"
                f"{result['Max3D']:.10f},"
                f"{result['ZRMSE']:.10f}"
                for name, result in results.items()
            ]
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
