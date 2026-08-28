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


METHODS = [
    (
        "GPS + IMU",
        os.path.join(
            ROOT,
            "results",
            "four_sensor_ablation",
            "gps_imu",
            "trajectory.txt",
        ),
    ),
    (
        "GPS + IMU + LiDAR",
        os.path.join(
            ROOT,
            "results",
            "four_sensor_ablation",
            "gps_imu_lidar",
            "trajectory.txt",
        ),
    ),
    (
        "GPS + IMU + Camera",
        os.path.join(
            ROOT,
            "results",
            "four_sensor_ablation",
            "gps_imu_camera",
            "trajectory.txt",
        ),
    ),
    (
        "GPS + IMU + LiDAR + Camera",
        os.path.join(
            ROOT,
            "results",
            "four_sensor_ablation",
            "gps_imu_lidar_camera",
            "trajectory.txt",
        ),
    ),
    (
        "Predictive Four-Sensor",
        os.path.join(
            ROOT,
            "results",
            "four_sensor_predictive_fg",
            "trajectory.txt",
        ),
    ),
]


REFERENCE = os.path.join(
    ROOT,
    "results",
    "four_sensor_fixed_fg",
    "gps_local_reference.txt",
)


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
    }


def main():
    if not Path(
        REFERENCE
    ).exists():
        raise FileNotFoundError(
            "Run run_four_sensor_fixed_fg.py first"
        )

    reference = np.loadtxt(
        REFERENCE,
        dtype=np.float64,
    )

    print("=" * 96)
    print(
        "FOUR-SENSOR FACTOR GRAPH EVALUATION"
    )
    print("=" * 96)

    for name, path in METHODS:
        if not Path(
            path
        ).exists():
            print(
                f"{name:32s}: NOT RUN"
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

        print(
            f"{name:32s} "
            f"ATE3D={result['ATE3D']:.6f} "
            f"ATE2D={result['ATE2D']:.6f} "
            f"Mean3D={result['Mean3D']:.6f} "
            f"Max3D={result['Max3D']:.6f}"
        )


if __name__ == "__main__":
    main()
