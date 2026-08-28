from __future__ import annotations

import os
import sys

import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(
        0,
        ROOT,
    )


from src.reliability.oracle_factor_reliability import (
    OracleReliabilityConfig,
    compute_oracle_factor_reliability,
)


DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

BODY_DIR = os.path.join(
    ROOT,
    "results",
    "body_relative_odometry",
)

CLEAN_POSES_PATH = os.path.join(
    ROOT,
    "results",
    "four_sensor_fixed_fg",
    "poses.npy",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "oracle_factor_reliability",
)


def finite_stats(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    return (
        float(
            np.min(
                values
            )
        ),
        float(
            np.max(
                values
            )
        ),
        float(
            np.mean(
                values
            )
        ),
        float(
            np.median(
                values
            )
        ),
    )


def print_stats(
    title,
    values,
):
    minimum, maximum, mean, median = finite_stats(
        values
    )

    print(
        f"{title:32s} "
        f"min={minimum:.6f} "
        f"max={maximum:.6f} "
        f"mean={mean:.6f} "
        f"median={median:.6f}"
    )


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    sensor = np.load(
        os.path.join(
            DEGRADED_DIR,
            "degraded_sensor_data.npz",
        ),
        allow_pickle=False,
    )

    lidar = np.load(
        os.path.join(
            DEGRADED_DIR,
            "degraded_lidar_factor_data.npz",
        ),
        allow_pickle=False,
    )

    camera = np.load(
        os.path.join(
            DEGRADED_DIR,
            "degraded_camera_factor_data.npz",
        ),
        allow_pickle=False,
    )

    reference_poses = np.load(
        CLEAN_POSES_PATH,
        allow_pickle=False,
    )

    config = OracleReliabilityConfig()

    result = compute_oracle_factor_reliability(
        clean_gps_local=
            sensor[
                "clean_gps_local"
            ],

        degraded_gps_local=
            sensor[
                "degraded_gps_local"
            ],

        degraded_imu_gyro=
            sensor[
                "degraded_imu_gyro"
            ],

        timestamps_seconds=
            sensor[
                "timestamps_seconds"
            ],

        lidar_between=
            lidar[
                "between_measurements"
            ],

        lidar_valid=
            lidar[
                "valid"
            ],

        camera_between=
            camera[
                "between_measurements"
            ],

        camera_valid=
            camera[
                "valid"
            ],

        reference_poses=
            reference_poses,

        config=
            config,
    )

    np.savez_compressed(
        os.path.join(
            OUTPUT_DIR,
            "oracle_factor_reliability.npz",
        ),
        **result,
    )

    for sensor_name in (
        "gps",
        "imu",
        "lidar",
        "camera",
    ):
        np.savetxt(
            os.path.join(
                OUTPUT_DIR,
                f"{sensor_name}_oracle_reliability.txt",
            ),
            result[
                f"{sensor_name}_reliability"
            ],
            fmt="%.8f",
        )

    print("=" * 108)
    print(
        "ORACLE FACTOR RELIABILITY V1"
    )
    print("=" * 108)

    print_stats(
        "GPS position error [m]",
        result[
            "gps_error_m"
        ],
    )

    print_stats(
        "GPS reliability",
        result[
            "gps_reliability"
        ],
    )

    print_stats(
        "IMU rotation error [deg]",
        result[
            "imu_rotation_error_deg"
        ],
    )

    print_stats(
        "IMU reliability",
        result[
            "imu_reliability"
        ],
    )

    print_stats(
        "LiDAR translation error [m]",
        result[
            "lidar_translation_error_m"
        ],
    )

    print_stats(
        "LiDAR rotation error [deg]",
        result[
            "lidar_rotation_error_deg"
        ],
    )

    print_stats(
        "LiDAR reliability",
        result[
            "lidar_reliability"
        ],
    )

    print_stats(
        "Camera translation error [m]",
        result[
            "camera_translation_error_m"
        ],
    )

    print_stats(
        "Camera rotation error [deg]",
        result[
            "camera_rotation_error_deg"
        ],
    )

    print_stats(
        "Camera reliability",
        result[
            "camera_reliability"
        ],
    )

    print()
    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
