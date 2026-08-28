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


DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
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
        "Old Severity-Predictive FG",
        os.path.join(
            ROOT,
            "results",
            "degraded_four_sensor_predictive_fg",
            "trajectory.txt",
        ),
    ),

    (
        "Predicted Factor Reliability FG",
        os.path.join(
            ROOT,
            "results",
            "predicted_factor_reliability_fg",
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


def metrics(
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

    error = (
        trajectory[
            :n
        ]
        -
        reference[
            :n
        ]
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
                        ** 2
                    )
                )
            ),

        "ATE2D":
            float(
                np.sqrt(
                    np.mean(
                        error2d
                        ** 2
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
                        ** 2
                    )
                )
            ),
    }


def main():
    sensor_path = os.path.join(
        DEGRADED_DIR,
        "degraded_sensor_data.npz",
    )

    if not Path(
        sensor_path
    ).exists():
        raise FileNotFoundError(
            sensor_path
        )

    sensor = np.load(
        sensor_path,
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
        "PREDICTIVE FACTOR RELIABILITY FINAL COMPARISON V1"
    )
    print("=" * 116)

    results = {}

    for name, path in METHODS:
        if not Path(
            path
        ).exists():
            print(
                f"{name:38s}: NOT RUN"
            )

            continue

        trajectory = np.loadtxt(
            path,
            dtype=np.float64,
        )

        result = metrics(
            reference,
            trajectory,
        )

        results[
            name
        ] = result

        print(
            f"{name:38s} "
            f"ATE3D={result['ATE3D']:.6f} "
            f"ATE2D={result['ATE2D']:.6f} "
            f"Mean3D={result['Mean3D']:.6f} "
            f"Max3D={result['Max3D']:.6f} "
            f"ZRMSE={result['ZRMSE']:.6f}"
        )

    fixed_name = (
        "Degraded Fixed Four-Sensor"
    )

    learned_name = (
        "Predicted Factor Reliability FG"
    )

    oracle_name = (
        "Oracle Factor Reliability"
    )

    if (
        fixed_name
        in results
        and
        learned_name
        in results
    ):
        fixed = results[
            fixed_name
        ][
            "ATE3D"
        ]

        learned = results[
            learned_name
        ][
            "ATE3D"
        ]

        improvement = (
            (
                fixed
                -
                learned
            )
            /
            max(
                fixed,
                1e-12,
            )
            *
            100.0
        )

        print()
        print(
            "Predicted factor-reliability improvement "
            f"vs fixed: {improvement:.2f}%"
        )

    if (
        learned_name
        in results
        and
        oracle_name
        in results
    ):
        learned = results[
            learned_name
        ][
            "ATE3D"
        ]

        oracle = results[
            oracle_name
        ][
            "ATE3D"
        ]

        gap = (
            (
                learned
                -
                oracle
            )
            /
            max(
                oracle,
                1e-12,
            )
            *
            100.0
        )

        print(
            "Gap from Oracle: "
            f"{gap:.2f}%"
        )


if __name__ == "__main__":
    main()
