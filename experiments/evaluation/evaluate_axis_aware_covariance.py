from __future__ import annotations

import csv
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


GROUND_TRUTH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)

AXIS_ROOT = os.path.join(
    ROOT,
    "results",
    "axis_aware_covariance_ablation",
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
        "All Predictive V1",
        os.path.join(
            ROOT,
            "results",
            "predicted_factor_reliability_fg",
            "trajectory.txt",
        ),
    ),
    (
        "GPS+IMU+Camera Predictive",
        os.path.join(
            ROOT,
            "results",
            "sensorwise_predictive_ablation",
            "gps_imu_camera",
            "trajectory.txt",
        ),
    ),
    (
        "Axis Control",
        os.path.join(
            AXIS_ROOT,
            "best_subset_control",
            "trajectory.txt",
        ),
    ),
    (
        "GPS Z Fixed",
        os.path.join(
            AXIS_ROOT,
            "gps_z_fixed",
            "trajectory.txt",
        ),
    ),
    (
        "Camera Z Fixed",
        os.path.join(
            AXIS_ROOT,
            "camera_z_fixed",
            "trajectory.txt",
        ),
    ),
    (
        "GPS + Camera Z Fixed",
        os.path.join(
            AXIS_ROOT,
            "gps_camera_z_fixed",
            "trajectory.txt",
        ),
    ),
    (
        "Camera Roll/Pitch/Z Fixed",
        os.path.join(
            AXIS_ROOT,
            "camera_rpz_fixed",
            "trajectory.txt",
        ),
    ),
    (
        "Full Axis-Aware",
        os.path.join(
            AXIS_ROOT,
            "full_axis_aware",
            "trajectory.txt",
        ),
    ),
    (
        "Error-derived Oracle All",
        os.path.join(
            ROOT,
            "results",
            "oracle_four_sensor_fg",
            "trajectory.txt",
        ),
    ),
]


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

    reference = np.asarray(
        reference[
            :n,
            :3
        ],
        dtype=np.float64,
    )

    trajectory = np.asarray(
        trajectory[
            :n,
            :3
        ],
        dtype=np.float64,
    )

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
    if not Path(
        GROUND_TRUTH
    ).exists():
        raise FileNotFoundError(
            GROUND_TRUTH
        )

    reference = np.loadtxt(
        GROUND_TRUTH,
        dtype=np.float64,
    )

    print(
        "=" * 128
    )

    print(
        "AXIS-AWARE / DOF-SPECIFIC DYNAMIC COVARIANCE EVALUATION"
    )

    print(
        "=" * 128
    )

    results = {}

    for name, path in METHODS:
        if not Path(
            path
        ).exists():
            print(
                f"{name:34s}: NOT RUN"
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

        print(
            f"{name:34s} "
            f"ATE3D={result['ATE3D']:.6f} "
            f"ATE2D={result['ATE2D']:.6f} "
            f"Mean3D={result['Mean3D']:.6f} "
            f"Max3D={result['Max3D']:.6f} "
            f"ZRMSE={result['ZRMSE']:.6f}"
        )

    baseline_name = (
        "GPS+IMU+Camera Predictive"
    )

    if baseline_name in results:
        baseline = results[
            baseline_name
        ]

        print()
        print(
            "-" * 128
        )

        print(
            "IMPROVEMENT VS CURRENT BEST "
            "(GPS+IMU+Camera Predictive)"
        )

        print(
            "-" * 128
        )

        for name in (
            "GPS Z Fixed",
            "Camera Z Fixed",
            "GPS + Camera Z Fixed",
            "Camera Roll/Pitch/Z Fixed",
            "Full Axis-Aware",
        ):
            if name not in results:
                continue

            current = results[
                name
            ]

            ate_gain = (
                baseline[
                    "ATE3D"
                ]
                -
                current[
                    "ATE3D"
                ]
            )

            z_gain = (
                baseline[
                    "ZRMSE"
                ]
                -
                current[
                    "ZRMSE"
                ]
            )

            ate_percent = (
                ate_gain
                /
                baseline[
                    "ATE3D"
                ]
                *
                100.0
            )

            z_percent = (
                z_gain
                /
                baseline[
                    "ZRMSE"
                ]
                *
                100.0
            )

            print(
                f"{name:34s} "
                f"ATE3D gain={ate_gain:+.6f} "
                f"({ate_percent:+.2f}%) | "
                f"ZRMSE gain={z_gain:+.6f} "
                f"({z_percent:+.2f}%)"
            )

    # Find the best axis-aware method by ATE3D.
    candidate_names = [
        name
        for name in (
            "GPS Z Fixed",
            "Camera Z Fixed",
            "GPS + Camera Z Fixed",
            "Camera Roll/Pitch/Z Fixed",
            "Full Axis-Aware",
        )
        if name in results
    ]

    if candidate_names:
        best_name = min(
            candidate_names,
            key=lambda name:
                results[
                    name
                ][
                    "ATE3D"
                ],
        )

        best = results[
            best_name
        ]

        print()
        print(
            "=" * 128
        )

        print(
            "BEST AXIS-AWARE METHOD:"
        )

        print(
            best_name
        )

        print(
            f"ATE3D={best['ATE3D']:.6f} "
            f"ATE2D={best['ATE2D']:.6f} "
            f"ZRMSE={best['ZRMSE']:.6f} "
            f"Max3D={best['Max3D']:.6f}"
        )

        print(
            "=" * 128
        )

    output_csv = os.path.join(
        ROOT,
        "results",
        "axis_aware_covariance_summary.csv",
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Method",
                "ATE3D",
                "ATE2D",
                "Mean3D",
                "Max3D",
                "ZRMSE",
            ],
        )

        writer.writeheader()

        for name, _ in METHODS:
            if name not in results:
                continue

            writer.writerow(
                {
                    "Method":
                        name,
                    **results[
                        name
                    ],
                }
            )

    print()
    print(
        "Saved:",
        output_csv,
    )


if __name__ == "__main__":
    main()
