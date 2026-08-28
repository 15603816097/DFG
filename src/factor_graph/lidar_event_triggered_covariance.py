from __future__ import annotations

from pathlib import Path

import numpy as np

from src.factor_graph.predictive_covariance import (
    reliability_to_sigma,
)

from src.factor_graph.sensor_factor_config import (
    FourSensorFactorConfig,
)


def lidar_event_sigma(
    config,
    trigger_active: bool,
    event_reliability: float,
):
    """
    Healthy:
        keep original fixed LiDAR covariance.

    Triggered:
        inflate LiDAR covariance using the EXISTING learned
        reliability-to-sigma mapping.

    This means the experiment changes LiDAR only on detected abnormal events.
    """
    if not bool(
        trigger_active
    ):
        return (
            float(
                config.lidar_rotation_sigma
            ),
            float(
                config.lidar_translation_sigma
            ),
        )

    rotation_sigma = float(
        reliability_to_sigma(
            event_reliability,
            config.lidar_rot_sigma_min,
            config.lidar_rot_sigma_max,
            config.lidar_gamma,
        )
    )

    translation_sigma = float(
        reliability_to_sigma(
            event_reliability,
            config.lidar_trans_sigma_min,
            config.lidar_trans_sigma_max,
            config.lidar_gamma,
        )
    )

    return (
        rotation_sigma,
        translation_sigma,
    )


def _timestamp_seconds(
    timestamps,
):
    if len(
        timestamps
    ) == 0:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    first = timestamps[
        0
    ]

    if hasattr(
        first,
        "timestamp",
    ):
        return np.asarray(
            [
                float(
                    t.timestamp()
                )
                for t in timestamps
            ],
            dtype=np.float64,
        )

    return np.asarray(
        timestamps,
        dtype=np.float64,
    )


def _pose3_from_matrix(
    gtsam,
    T,
):
    T = np.asarray(
        T,
        dtype=np.float64,
    )

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


def _pose3_to_matrix(
    pose,
):
    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[
        :3,
        :3
    ] = (
        pose.rotation().matrix()
    )

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


def _imu_between_pose(
    gtsam,
    gyro_t,
    dt,
):
    gyro_t = np.asarray(
        gyro_t,
        dtype=np.float64,
    )

    rotvec = (
        gyro_t
        *
        float(
            dt
        )
    )

    return gtsam.Pose3(
        gtsam.Rot3.Expmap(
            rotvec
        ),
        gtsam.Point3(
            0.0,
            0.0,
            0.0,
        ),
    )


def _initial_trajectory(
    gtsam,
    measurements,
):
    n = len(
        measurements.gps_local
    )

    poses = [
        gtsam.Pose3()
    ]

    for i in range(
        n
        -
        1
    ):
        if bool(
            measurements.lidar_valid[
                i
            ]
        ):
            delta = _pose3_from_matrix(
                gtsam,
                measurements.lidar_between[
                    i
                ],
            )

        elif bool(
            measurements.camera_valid[
                i
            ]
        ):
            delta = _pose3_from_matrix(
                gtsam,
                measurements.camera_between[
                    i
                ],
            )

        else:
            delta = gtsam.Pose3()

        poses.append(
            poses[
                -1
            ].compose(
                delta
            )
        )

    return poses


def _load_prediction(
    prediction_dir,
    sensor,
    n,
):
    path = (
        Path(
            prediction_dir
        )
        /
        f"{sensor}_predictive_prior_target_aligned.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    values = np.loadtxt(
        path,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if len(
        values
    ) < n:
        raise ValueError(
            f"{path}: {len(values)} values, "
            f"expected at least {n}"
        )

    return np.clip(
        values[
            :n
        ],
        0.0,
        1.0,
    )


def _dynamic_sigma(
    reliability,
    sigma_min,
    sigma_max,
    gamma,
):
    return float(
        reliability_to_sigma(
            reliability,
            sigma_min,
            sigma_max,
            gamma,
        )
    )


def run_lidar_event_triggered_factor_graph(
    measurements,
    output_dir,
    prediction_dir,
    event_data_path,
    config=None,
):
    """
    Final graph structure for this experiment:

        GPS     predictive
        IMU     predictive
        Camera  predictive
        LiDAR   fixed by default
                dynamic ONLY on event-triggered frames
    """
    try:
        import gtsam

    except Exception as exc:
        raise ImportError(
            "gtsam is required"
        ) from exc

    if config is None:
        config = (
            FourSensorFactorConfig()
        )

    n = len(
        measurements.gps_local
    )

    predictions = {
        sensor:
            _load_prediction(
                prediction_dir,
                sensor,
                n,
            )
        for sensor in (
            "gps",
            "imu",
            "camera",
        )
    }

    event_data = np.load(
        event_data_path,
        allow_pickle=False,
    )

    trigger_mask = np.asarray(
        event_data[
            "trigger_mask"
        ],
        dtype=bool,
    ).reshape(
        -1
    )

    event_reliability = np.asarray(
        event_data[
            "event_reliability"
        ],
        dtype=np.float64,
    ).reshape(
        -1
    )

    if len(
        trigger_mask
    ) < n:
        raise ValueError(
            "LiDAR event trigger length "
            f"{len(trigger_mask)} < {n}"
        )

    trigger_mask = trigger_mask[
        :n
    ]

    event_reliability = (
        event_reliability[
            :n
        ]
    )

    graph = (
        gtsam.NonlinearFactorGraph()
    )

    initial = (
        gtsam.Values()
    )

    poses0 = _initial_trajectory(
        gtsam,
        measurements,
    )

    prior_noise = (
        gtsam.noiseModel.Diagonal.Sigmas(
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
    )

    graph.add(
        gtsam.PriorFactorPose3(
            0,
            gtsam.Pose3(),
            prior_noise,
        )
    )

    for i in range(
        n
    ):
        initial.insert(
            i,
            poses0[
                i
            ],
        )

    times = _timestamp_seconds(
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

    triggered_lidar_factors = 0

    lidar_rotation_sigmas = []
    lidar_translation_sigmas = []

    for i in range(
        n
    ):
        # ----------------------------------------------------------
        # GPS predictive
        # ----------------------------------------------------------
        gps_r = predictions[
            "gps"
        ][
            i
        ]

        gps_sigma = _dynamic_sigma(
            gps_r,
            config.gps_sigma_min,
            config.gps_sigma_max,
            config.gps_gamma,
        )

        gps_noise = (
            gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    [
                        1e6,
                        1e6,
                        1e6,
                        gps_sigma,
                        gps_sigma,
                        gps_sigma,
                    ],
                    dtype=np.float64,
                )
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

        dt = (
            float(
                times[
                    i + 1
                ]
                -
                times[
                    i
                ]
            )
            if len(
                times
            )
            ==
            n
            else
            0.1
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
        # IMU predictive
        # ----------------------------------------------------------
        imu_r = predictions[
            "imu"
        ][
            i + 1
        ]

        imu_sigma = _dynamic_sigma(
            imu_r,
            config.imu_rot_sigma_min,
            config.imu_rot_sigma_max,
            config.imu_gamma,
        )

        imu_noise = (
            gtsam.noiseModel.Diagonal.Sigmas(
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
        )

        graph.add(
            gtsam.BetweenFactorPose3(
                i,
                i + 1,
                _imu_between_pose(
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
        # LiDAR:
        # fixed unless event trigger is active.
        # ----------------------------------------------------------
        if (
            bool(
                measurements.lidar_valid[
                    i
                ]
            )
            and
            float(
                measurements.lidar_quality[
                    i
                ]
            )
            >=
            config.lidar_quality_min
        ):
            target_frame = (
                i
                +
                1
            )

            active = bool(
                trigger_mask[
                    target_frame
                ]
            )

            (
                lidar_rot_sigma,
                lidar_trans_sigma,
            ) = lidar_event_sigma(
                config,
                trigger_active=
                    active,
                event_reliability=
                    event_reliability[
                        target_frame
                    ],
            )

            if active:
                triggered_lidar_factors += 1

            lidar_rotation_sigmas.append(
                lidar_rot_sigma
            )

            lidar_translation_sigmas.append(
                lidar_trans_sigma
            )

            lidar_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
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
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    _pose3_from_matrix(
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
        # Camera predictive
        # ----------------------------------------------------------
        if (
            bool(
                measurements.camera_valid[
                    i
                ]
            )
            and
            float(
                measurements.camera_quality[
                    i
                ]
            )
            >=
            config.camera_quality_min
        ):
            camera_r = predictions[
                "camera"
            ][
                i + 1
            ]

            camera_rot = _dynamic_sigma(
                camera_r,
                config.camera_rot_sigma_min,
                config.camera_rot_sigma_max,
                config.camera_gamma,
            )

            camera_trans = _dynamic_sigma(
                camera_r,
                config.camera_trans_sigma_min,
                config.camera_trans_sigma_max,
                config.camera_gamma,
            )

            camera_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.asarray(
                        [
                            camera_rot,
                            camera_rot,
                            camera_rot,
                            camera_trans,
                            camera_trans,
                            camera_trans,
                        ],
                        dtype=np.float64,
                    )
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    _pose3_from_matrix(
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

    print(
        "Triggered LiDAR factors:",
        triggered_lidar_factors,
        "/",
        counts[
            "lidar"
        ],
    )

    if lidar_translation_sigmas:
        trans = np.asarray(
            lidar_translation_sigmas,
            dtype=np.float64,
        )

        rot = np.asarray(
            lidar_rotation_sigmas,
            dtype=np.float64,
        )

        print(
            "LiDAR translation sigma "
            "min/max/mean:",
            float(
                trans.min()
            ),
            float(
                trans.max()
            ),
            float(
                trans.mean()
            ),
        )

        print(
            "LiDAR rotation sigma "
            "min/max/mean:",
            float(
                rot.min()
            ),
            float(
                rot.max()
            ),
            float(
                rot.mean()
            ),
        )

    params = (
        gtsam.LevenbergMarquardtParams()
    )

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

        T = _pose3_to_matrix(
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

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        output_dir
        /
        "trajectory.txt",
        trajectory,
        fmt="%.8f",
    )

    np.save(
        output_dir
        /
        "poses.npy",
        poses,
    )

    return (
        trajectory,
        poses,
        counts,
        triggered_lidar_factors,
    )
