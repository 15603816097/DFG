from __future__ import annotations

import os
import sys
from pathlib import Path

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


from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)

from src.factor_graph.sensor_factor_config import (
    FourSensorFactorConfig,
)

from src.factor_graph.predictive_covariance import (
    reliability_to_sigma,
    combine_prediction_with_measurement_quality,
)


DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

ORACLE_DIR = os.path.join(
    ROOT,
    "results",
    "oracle_factor_reliability",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "oracle_four_sensor_fg",
)


def pose3_from_matrix(
    gtsam,
    T,
):
    return gtsam.Pose3(
        gtsam.Rot3(
            T[
                :3,
                :3
            ]
        ),
        gtsam.Point3(
            float(
                T[
                    0,
                    3
                ]
            ),
            float(
                T[
                    1,
                    3
                ]
            ),
            float(
                T[
                    2,
                    3
                ]
            ),
        ),
    )


def pose3_to_matrix(
    pose,
):
    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[
        :3,
        :3
    ] = pose.rotation().matrix()

    T[
        :3,
        3
    ] = np.asarray(
        pose.translation(),
        dtype=np.float64,
    ).reshape(
        3
    )

    return T


def timestamp_seconds(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return values


def imu_between_pose(
    gtsam,
    gyro_t,
    dt,
):
    omega = np.asarray(
        gyro_t,
        dtype=np.float64,
    )

    rotvec = (
        omega
        *
        float(
            dt
        )
    )

    R = gtsam.Rot3.Expmap(
        rotvec
    )

    return gtsam.Pose3(
        R,
        gtsam.Point3(
            0.0,
            0.0,
            0.0,
        ),
    )


def main():
    try:
        import gtsam
    except Exception as exc:
        raise ImportError(
            "gtsam is required"
        ) from exc

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    measurements = (
        load_degraded_four_sensor_measurements(
            DEGRADED_DIR
        )
    )

    oracle = np.load(
        os.path.join(
            ORACLE_DIR,
            "oracle_factor_reliability.npz",
        ),
        allow_pickle=False,
    )

    config = FourSensorFactorConfig()

    n = len(
        measurements.gps_local
    )

    graph = gtsam.NonlinearFactorGraph()

    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(
            [
                config.prior_rotation_sigma,
                config.prior_rotation_sigma,
                config.prior_rotation_sigma,
                config.prior_translation_sigma,
                config.prior_translation_sigma,
                config.prior_translation_sigma,
            ],
            dtype=np.float64,
        )
    )

    graph.add(
        gtsam.PriorFactorPose3(
            0,
            gtsam.Pose3(),
            prior_noise,
        )
    )

    # Initialize from degraded LiDAR, with camera fallback.
    pose = gtsam.Pose3()

    initial.insert(
        0,
        pose,
    )

    for i in range(
        n - 1
    ):
        if measurements.lidar_valid[
            i
        ]:
            delta = pose3_from_matrix(
                gtsam,
                measurements.lidar_between[
                    i
                ],
            )

        elif measurements.camera_valid[
            i
        ]:
            delta = pose3_from_matrix(
                gtsam,
                measurements.camera_between[
                    i
                ],
            )

        else:
            delta = gtsam.Pose3()

        pose = pose.compose(
            delta
        )

        initial.insert(
            i + 1,
            pose,
        )

    times = timestamp_seconds(
        measurements.timestamps
    )

    counts = {
        "gps":
            0,
        "imu":
            0,
        "lidar":
            0,
        "camera":
            0,
    }

    for i in range(
        n
    ):
        # ----------------------------------------------------------
        # GPS oracle covariance
        # ----------------------------------------------------------
        sigma_gps = float(
            reliability_to_sigma(
                oracle[
                    "gps_reliability"
                ][
                    i
                ],
                config.gps_sigma_min,
                config.gps_sigma_max,
                config.gps_gamma,
            )
        )

        gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    1e6,
                    1e6,
                    1e6,
                    sigma_gps,
                    sigma_gps,
                    sigma_gps,
                ],
                dtype=np.float64,
            )
        )

        gps_pose = gtsam.Pose3(
            gtsam.Rot3(),
            gtsam.Point3(
                float(
                    measurements.gps_local[
                        i,
                        0
                    ]
                ),
                float(
                    measurements.gps_local[
                        i,
                        1
                    ]
                ),
                float(
                    measurements.gps_local[
                        i,
                        2
                    ]
                ),
            ),
        )

        graph.add(
            gtsam.PriorFactorPose3(
                i,
                gps_pose,
                gps_noise,
            )
        )

        counts[
            "gps"
        ] += 1

        if i >= n - 1:
            continue

        dt = float(
            times[
                i + 1
            ]
            -
            times[
                i
            ]
        )

        if (
            not np.isfinite(
                dt
            )
            or
            dt <= 0.0
            or
            dt > 1.0
        ):
            dt = 0.1

        # ----------------------------------------------------------
        # IMU oracle covariance
        # ----------------------------------------------------------
        imu_sigma = float(
            reliability_to_sigma(
                oracle[
                    "imu_reliability"
                ][
                    i + 1
                ],
                config.imu_rot_sigma_min,
                config.imu_rot_sigma_max,
                config.imu_gamma,
            )
        )

        imu_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    imu_sigma,
                    imu_sigma,
                    imu_sigma,
                    config.imu_translation_sigma,
                    config.imu_translation_sigma,
                    config.imu_translation_sigma,
                ],
                dtype=np.float64,
            )
        )

        graph.add(
            gtsam.BetweenFactorPose3(
                i,
                i + 1,
                imu_between_pose(
                    gtsam,
                    measurements.imu_gyro[
                        i
                    ],
                    dt,
                ),
                imu_noise,
            )
        )

        counts[
            "imu"
        ] += 1

        # ----------------------------------------------------------
        # LiDAR oracle covariance
        # ----------------------------------------------------------
        if measurements.lidar_valid[
            i
        ]:
            effective = float(
                combine_prediction_with_measurement_quality(
                    oracle[
                        "lidar_reliability"
                    ][
                        i + 1
                    ],
                    measurements.lidar_quality[
                        i
                    ],
                )
            )

            lidar_rot_sigma = float(
                reliability_to_sigma(
                    effective,
                    config.lidar_rot_sigma_min,
                    config.lidar_rot_sigma_max,
                    config.lidar_gamma,
                )
            )

            lidar_trans_sigma = float(
                reliability_to_sigma(
                    effective,
                    config.lidar_trans_sigma_min,
                    config.lidar_trans_sigma_max,
                    config.lidar_gamma,
                )
            )

            lidar_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    [
                        lidar_rot_sigma,
                        lidar_rot_sigma,
                        lidar_rot_sigma,
                        lidar_trans_sigma,
                        lidar_trans_sigma,
                        lidar_trans_sigma,
                    ],
                    dtype=np.float64,
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    pose3_from_matrix(
                        gtsam,
                        measurements.lidar_between[
                            i
                        ],
                    ),
                    lidar_noise,
                )
            )

            counts[
                "lidar"
            ] += 1

        # ----------------------------------------------------------
        # Camera oracle covariance
        # ----------------------------------------------------------
        if measurements.camera_valid[
            i
        ]:
            effective = float(
                combine_prediction_with_measurement_quality(
                    oracle[
                        "camera_reliability"
                    ][
                        i + 1
                    ],
                    measurements.camera_quality[
                        i
                    ],
                )
            )

            camera_rot_sigma = float(
                reliability_to_sigma(
                    effective,
                    config.camera_rot_sigma_min,
                    config.camera_rot_sigma_max,
                    config.camera_gamma,
                )
            )

            camera_trans_sigma = float(
                reliability_to_sigma(
                    effective,
                    config.camera_trans_sigma_min,
                    config.camera_trans_sigma_max,
                    config.camera_gamma,
                )
            )

            camera_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    [
                        camera_rot_sigma,
                        camera_rot_sigma,
                        camera_rot_sigma,
                        camera_trans_sigma,
                        camera_trans_sigma,
                        camera_trans_sigma,
                    ],
                    dtype=np.float64,
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    pose3_from_matrix(
                        gtsam,
                        measurements.camera_between[
                            i
                        ],
                    ),
                    camera_noise,
                )
            )

            counts[
                "camera"
            ] += 1

        if i % 500 == 0:
            print(
                "Add factors:",
                i,
            )

    print(
        "Graph factors:",
        graph.size(),
    )

    print(
        "Factor counts:",
        counts,
    )

    params = gtsam.LevenbergMarquardtParams()

    params.setMaxIterations(
        100
    )

    params.setRelativeErrorTol(
        1e-7
    )

    print(
        "Optimizing..."
    )

    result = (
        gtsam.LevenbergMarquardtOptimizer(
            graph,
            initial,
            params,
        ).optimize()
    )

    trajectory = np.zeros(
        (
            n,
            3,
        ),
        dtype=np.float64,
    )

    poses = np.zeros(
        (
            n,
            4,
            4,
        ),
        dtype=np.float64,
    )

    for i in range(
        n
    ):
        pose = result.atPose3(
            i
        )

        T = pose3_to_matrix(
            pose
        )

        poses[
            i
        ] = T

        trajectory[
            i
        ] = T[
            :3,
            3
        ]

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "trajectory.txt",
        ),
        trajectory,
        fmt="%.8f",
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "poses.npy",
        ),
        poses,
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "factor_counts.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        for key, value in counts.items():
            file.write(
                f"{key}={value}\n"
            )

    print(
        "Trajectory:",
        trajectory.shape,
    )

    print(
        "Final position:",
        trajectory[
            -1
        ],
    )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
