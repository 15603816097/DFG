"""
Build balanced multi-sensor reliability training data V2.

Uses the already implemented corruption functions and feature extraction.
Only the degradation plan changes.
"""

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


from src.loader.gps_loader import GPSLoader
from src.loader.imu_loader import IMULoader
from src.loader.lidar_loader import LidarLoader
from src.loader.camera_loader import CameraLoader

from src.multisensor_reliability.config import (
    MultiSensorReliabilityConfig,
)

from src.multisensor_reliability.degradation.balanced_plans import (
    load_balanced_plan,
)

from src.multisensor_reliability.degradation.apply import (
    corrupt_gps,
    corrupt_imu,
    corrupt_lidar,
    corrupt_camera,
)

from src.multisensor_reliability.labels import (
    build_future_labels,
)

from src.multisensor_reliability.features import (
    gps_features,
    imu_features,
    lidar_features,
    camera_features,
)


DATASET = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)

PLAN_DIR = os.path.join(
    ROOT,
    "results",
    "multisensor_reliability_v2",
)

OUTPUT_PATH = os.path.join(
    PLAN_DIR,
    "multisensor_reliability_data_v2.npz",
)


def extract_lidar(
    item,
):
    if isinstance(
        item,
        np.ndarray,
    ):
        return item

    if isinstance(
        item,
        dict,
    ):
        for key in (
            "points",
            "point_cloud",
            "lidar",
            "data",
        ):
            if key in item:
                return np.asarray(
                    item[
                        key
                    ]
                )

    raise KeyError(
        "Cannot extract LiDAR points. "
        "Check LidarLoader output keys."
    )


def extract_camera(
    item,
):
    if isinstance(
        item,
        np.ndarray,
    ):
        return item

    if isinstance(
        item,
        dict,
    ):
        for key in (
            "image",
            "rgb",
            "data",
        ):
            if key in item:
                return np.asarray(
                    item[
                        key
                    ]
                )

    raise KeyError(
        "Cannot extract camera image. "
        "Check CameraLoader output keys."
    )


def main():
    config = (
        MultiSensorReliabilityConfig()
    )

    plan = load_balanced_plan(
        PLAN_DIR
    )

    gps_loader = GPSLoader(
        DATASET
    )

    imu_loader = IMULoader(
        DATASET
    )

    lidar_loader = LidarLoader(
        DATASET
    )

    camera_loader = CameraLoader(
        DATASET,
        camera_id=
            "image_02",
    )

    n = min(
        len(
            gps_loader
        ),
        len(
            imu_loader
        ),
        len(
            lidar_loader
        ),
        len(
            camera_loader
        ),
        int(
            plan[
                "n_frames"
            ]
        ),
    )

    gps_rows = []
    imu_acc_rows = []
    imu_gyro_rows = []
    lidar_rows = []
    camera_rows = []

    seed = int(
        plan[
            "seed"
        ]
    )

    print("=" * 92)
    print(
        "BUILD BALANCED MULTI-SENSOR "
        "RELIABILITY DATA V2"
    )
    print("=" * 92)

    print(
        "Frames:",
        n,
    )

    for i in range(n):
        gps_item = gps_loader[
            i
        ]

        imu_item = imu_loader[
            i
        ]

        lidar_item = lidar_loader[
            i
        ]

        camera_item = camera_loader[
            i
        ]

        gps_clean = np.asarray(
            gps_item[
                "position"
            ],
            dtype=np.float64,
        )

        acc_clean = np.asarray(
            imu_item[
                "acceleration"
            ],
            dtype=np.float64,
        )

        gyro_clean = np.asarray(
            imu_item[
                "angular_velocity"
            ],
            dtype=np.float64,
        )

        lidar_clean = extract_lidar(
            lidar_item
        )

        camera_clean = extract_camera(
            camera_item
        )

        gps_corrupted = corrupt_gps(
            gps_clean,
            plan[
                "severity"
            ][
                "gps"
            ][i],
            plan[
                "mode"
            ][
                "gps"
            ][i],
            i,
            seed=seed,
        )

        (
            acc_corrupted,
            gyro_corrupted,
        ) = corrupt_imu(
            acc_clean,
            gyro_clean,
            plan[
                "severity"
            ][
                "imu"
            ][i],
            plan[
                "mode"
            ][
                "imu"
            ][i],
            i,
            seed=seed,
        )

        lidar_corrupted = (
            corrupt_lidar(
                lidar_clean,
                plan[
                    "severity"
                ][
                    "lidar"
                ][i],
                plan[
                    "mode"
                ][
                    "lidar"
                ][i],
                i,
                seed=seed,
            )
        )

        camera_corrupted = (
            corrupt_camera(
                camera_clean,
                plan[
                    "severity"
                ][
                    "camera"
                ][i],
                plan[
                    "mode"
                ][
                    "camera"
                ][i],
                i,
                seed=seed,
            )
        )

        gps_rows.append(
            gps_corrupted
        )

        imu_acc_rows.append(
            acc_corrupted
        )

        imu_gyro_rows.append(
            gyro_corrupted
        )

        lidar_rows.append(
            lidar_features(
                lidar_corrupted
            )
        )

        camera_rows.append(
            camera_features(
                camera_corrupted
            )
        )

        if i % 500 == 0:
            print(
                "Processed:",
                i,
            )

    gps_rows = np.asarray(
        gps_rows,
        dtype=np.float64,
    )

    imu_acc_rows = np.asarray(
        imu_acc_rows,
        dtype=np.float64,
    )

    imu_gyro_rows = np.asarray(
        imu_gyro_rows,
        dtype=np.float64,
    )

    gps_feature_array = (
        gps_features(
            gps_rows,
            dt=
                config.dt,
            window=
                config.gps_window,
        )
    )

    imu_feature_array = (
        imu_features(
            imu_acc_rows,
            imu_gyro_rows,
            window=
                config.imu_window,
        )
    )

    lidar_feature_array = (
        np.asarray(
            lidar_rows,
            dtype=np.float32,
        )
    )

    camera_feature_array = (
        np.asarray(
            camera_rows,
            dtype=np.float32,
        )
    )

    current, future = (
        build_future_labels(
            {
                sensor:
                    plan[
                        "severity"
                    ][
                        sensor
                    ][
                        :n
                    ]
                for sensor in (
                    "gps",
                    "imu",
                    "lidar",
                    "camera",
                )
            },
            horizon=
                config.horizon,
            gamma=
                config.reliability_gamma,
        )
    )

    np.savez_compressed(
        OUTPUT_PATH,

        gps_features=
            gps_feature_array,

        imu_features=
            imu_feature_array,

        lidar_features=
            lidar_feature_array,

        camera_features=
            camera_feature_array,

        gps_current_label=
            current[
                "gps"
            ],

        imu_current_label=
            current[
                "imu"
            ],

        lidar_current_label=
            current[
                "lidar"
            ],

        camera_current_label=
            current[
                "camera"
            ],

        gps_future_label=
            future[
                "gps"
            ],

        imu_future_label=
            future[
                "imu"
            ],

        lidar_future_label=
            future[
                "lidar"
            ],

        camera_future_label=
            future[
                "camera"
            ],

        gps_severity=
            plan[
                "severity"
            ][
                "gps"
            ][
                :n
            ],

        imu_severity=
            plan[
                "severity"
            ][
                "imu"
            ][
                :n
            ],

        lidar_severity=
            plan[
                "severity"
            ][
                "lidar"
            ][
                :n
            ],

        camera_severity=
            plan[
                "severity"
            ][
                "camera"
            ][
                :n
            ],

        horizon=np.asarray(
            [
                config.horizon
            ],
            dtype=np.int64,
        ),
    )

    print()
    print(
        "GPS features:",
        gps_feature_array.shape,
    )

    print(
        "IMU features:",
        imu_feature_array.shape,
    )

    print(
        "LiDAR features:",
        lidar_feature_array.shape,
    )

    print(
        "Camera features:",
        camera_feature_array.shape,
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
