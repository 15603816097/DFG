from __future__ import annotations

import os
from pathlib import Path

import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


DEGRADED_SENSOR_PATH = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
    "degraded_sensor_data.npz",
)


METHODS = [
    (
        "Degraded Fixed Four-Sensor",
        os.path.join(
            ROOT,
            "results",
            "degraded_four_sensor_fixed_fg",
            "trajectory.txt",
        ),
    ),

    (
        "Predicted Factor Reliability V1",
        os.path.join(
            ROOT,
            "results",
            "predicted_factor_reliability_fg",
            "trajectory.txt",
        ),
    ),

    (
        "Conservative Predictive FG",
        os.path.join(
            ROOT,
            "results",
            "conservative_predictive_four_sensor_fg",
            "trajectory.txt",
        ),
    ),

    (
        "Oracle Factor Reliability",
        os.path.join(
            ROOT,
            "results",
            "oracle_four_sensor_fg",
            "trajectory.txt",
        ),
    ),
]


OUTPUT_PATH = os.path.join(
    ROOT,
    "results",
    "conservative_predictive_fg_evaluation.txt",
)


def calculate_metrics(
    reference,
    trajectory,
):
    n = min(
        len(
            reference
        ),
        len(
            trajectory
        ),
    )

    reference = reference[
        :n
    ]

    trajectory = trajectory[
        :n
    ]

    error = (
        trajectory
        -
        reference
    )

    error3d = np.linalg.norm(
        error,
        axis=1,
    )

    error2d = np.linalg.norm(
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
                        error3d
                        **
                        2
                    )
                )
            ),

        "ATE2D":
            float(
                np.sqrt(
                    np.mean(
                        error2d
                        **
                        2
                    )
                )
            ),

        "Mean3D":
            float(
                np.mean(
                    error3d
                )
            ),

        "Max3D":
            float(
                np.max(
                    error3d
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
    sensor = np.load(
        DEGRADED_SENSOR_PATH,
        allow_pickle=False,
    )

    reference = np.asarray(
        sensor[
            "clean_gps_local"
        ],
        dtype=np.float64,
    )

    print("=" * 116)
    print(
        "CONSERVATIVE PREDICTIVE FACTOR GRAPH EVALUATION"
    )
    print("=" * 116)

    results = {}

    lines = []

    for name, path in METHODS:
        if not Path(
            path
        ).exists():
            print(
                f"{name:40s}: NOT RUN"
            )

            continue

        trajectory = np.loadtxt(
            path,
            dtype=np.float64,
        )

        result = calculate_metrics(
            reference,
            trajectory,
        )

        results[
            name
        ] = result

        line = (
            f"{name:40s} "
            f"ATE3D={result['ATE3D']:.6f} "
            f"ATE2D={result['ATE2D']:.6f} "
            f"Mean3D={result['Mean3D']:.6f} "
            f"Max3D={result['Max3D']:.6f} "
            f"ZRMSE={result['ZRMSE']:.6f}"
        )

        print(
            line
        )

        lines.append(
            line
        )

    fixed = results.get(
        "Degraded Fixed Four-Sensor"
    )

    v1 = results.get(
        "Predicted Factor Reliability V1"
    )

    conservative = results.get(
        "Conservative Predictive FG"
    )

    oracle = results.get(
        "Oracle Factor Reliability"
    )

    print()

    if (
        fixed
        and
        conservative
    ):
        improvement = (
            (
                fixed[
                    "ATE3D"
                ]
                -
                conservative[
                    "ATE3D"
                ]
            )
            /
            fixed[
                "ATE3D"
            ]
            *
            100.0
        )

        text = (
            "Conservative improvement vs Fixed: "
            f"{improvement:.2f}%"
        )

        print(
            text
        )

        lines.append(
            text
        )

    if (
        v1
        and
        conservative
    ):
        improvement = (
            (
                v1[
                    "ATE3D"
                ]
                -
                conservative[
                    "ATE3D"
                ]
            )
            /
            v1[
                "ATE3D"
            ]
            *
            100.0
        )

        text = (
            "Conservative improvement vs V1: "
            f"{improvement:.2f}%"
        )

        print(
            text
        )

        lines.append(
            text
        )

    if (
        oracle
        and
        conservative
    ):
        gap = (
            (
                conservative[
                    "ATE3D"
                ]
                -
                oracle[
                    "ATE3D"
                ]
            )
            /
            oracle[
                "ATE3D"
            ]
            *
            100.0
        )

        text = (
            "Conservative gap from Oracle: "
            f"{gap:.2f}%"
        )

        print(
            text
        )

        lines.append(
            text
        )

    Path(
        OUTPUT_PATH
    ).write_text(
        "\n".join(
            lines
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
