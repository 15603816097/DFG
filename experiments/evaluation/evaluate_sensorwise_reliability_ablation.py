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


REFERENCE_PATH = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
    "degraded_sensor_data.npz",
)


PREDICTIVE_CASES = [
    "fixed",
    "gps_only",
    "imu_only",
    "lidar_only",
    "camera_only",
    "gps_imu",
    "gps_lidar",
    "gps_camera",
    "gps_imu_lidar",
    "gps_imu_camera",
    "all_predictive_v1",
]


ORACLE_CASES = [
    "gps_only",
    "imu_only",
    "lidar_only",
    "camera_only",
    "gps_imu",
    "gps_lidar",
    "gps_camera",
    "all_oracle",
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


def load_result(
    root,
    case,
):
    path = os.path.join(
        root,
        case,
        "trajectory.txt",
    )

    if not Path(
        path
    ).exists():
        return None

    return np.loadtxt(
        path,
        dtype=np.float64,
    )


def main():
    sensor = np.load(
        REFERENCE_PATH,
        allow_pickle=False,
    )

    reference = np.asarray(
        sensor[
            "clean_gps_local"
        ],
        dtype=np.float64,
    )

    predictive_root = os.path.join(
        ROOT,
        "results",
        "sensorwise_predictive_ablation",
    )

    oracle_root = os.path.join(
        ROOT,
        "results",
        "sensorwise_oracle_ablation",
    )

    rows = []

    print("=" * 124)
    print(
        "SENSOR-WISE RELIABILITY ABLATION"
    )
    print("=" * 124)

    print()
    print(
        "PREDICTIVE V1"
    )

    print(
        "-" * 124
    )

    fixed_ate = None

    predictive_metrics = {}

    for case in PREDICTIVE_CASES:
        trajectory = load_result(
            predictive_root,
            case,
        )

        if trajectory is None:
            print(
                f"{case:24s}: NOT RUN"
            )

            continue

        result = metrics(
            reference,
            trajectory,
        )

        predictive_metrics[
            case
        ] = result

        if case == "fixed":
            fixed_ate = result[
                "ATE3D"
            ]

        improvement = (
            0.0
            if fixed_ate is None
            else
            (
                fixed_ate
                -
                result[
                    "ATE3D"
                ]
            )
            /
            fixed_ate
            *
            100.0
        )

        print(
            f"{case:24s} "
            f"ATE3D={result['ATE3D']:.6f} "
            f"ATE2D={result['ATE2D']:.6f} "
            f"Max3D={result['Max3D']:.6f} "
            f"vsFixed={improvement:+.2f}%"
        )

        rows.append(
            {
                "family":
                    "predictive",
                "case":
                    case,
                **result,
                "vsFixedPercent":
                    improvement,
            }
        )

    print()
    print(
        "ORACLE"
    )

    print(
        "-" * 124
    )

    oracle_metrics = {}

    for case in ORACLE_CASES:
        trajectory = load_result(
            oracle_root,
            case,
        )

        if trajectory is None:
            print(
                f"{case:24s}: NOT RUN"
            )

            continue

        result = metrics(
            reference,
            trajectory,
        )

        oracle_metrics[
            case
        ] = result

        improvement = (
            float(
                "nan"
            )
            if fixed_ate is None
            else
            (
                fixed_ate
                -
                result[
                    "ATE3D"
                ]
            )
            /
            fixed_ate
            *
            100.0
        )

        print(
            f"{case:24s} "
            f"ATE3D={result['ATE3D']:.6f} "
            f"ATE2D={result['ATE2D']:.6f} "
            f"Max3D={result['Max3D']:.6f} "
            f"vsFixed={improvement:+.2f}%"
        )

        rows.append(
            {
                "family":
                    "oracle",
                "case":
                    case,
                **result,
                "vsFixedPercent":
                    improvement,
            }
        )

    print()
    print(
        "=" * 124
    )

    print(
        "SINGLE-SENSOR DIAGNOSTIC"
    )

    print(
        "=" * 124
    )

    for sensor_name in (
        "gps",
        "imu",
        "lidar",
        "camera",
    ):
        case = (
            f"{sensor_name}_only"
        )

        learned = predictive_metrics.get(
            case
        )

        oracle = oracle_metrics.get(
            case
        )

        if (
            learned is None
            or
            oracle is None
            or
            fixed_ate is None
        ):
            continue

        learned_gain = (
            fixed_ate
            -
            learned[
                "ATE3D"
            ]
        )

        oracle_gain = (
            fixed_ate
            -
            oracle[
                "ATE3D"
            ]
        )

        if oracle_gain > 1e-6:
            capture_ratio = (
                learned_gain
                /
                oracle_gain
            )
        else:
            capture_ratio = float(
                "nan"
            )

        print(
            f"{sensor_name:8s} "
            f"PredictiveATE={learned['ATE3D']:.6f} "
            f"OracleATE={oracle['ATE3D']:.6f} "
            f"LearnedGain={learned_gain:+.6f} "
            f"OracleGain={oracle_gain:+.6f} "
            f"CaptureRatio={capture_ratio:+.3f}"
        )

    output_csv = os.path.join(
        ROOT,
        "results",
        "sensorwise_reliability_ablation_summary.csv",
    )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "family",
            "case",
            "ATE3D",
            "ATE2D",
            "Mean3D",
            "Max3D",
            "ZRMSE",
            "vsFixedPercent",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        "Saved:",
        output_csv,
    )


if __name__ == "__main__":
    main()
