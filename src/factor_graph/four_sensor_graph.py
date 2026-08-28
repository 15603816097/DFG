from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.factor_graph.sensor_factor_config import (
    FourSensorFactorConfig,
)
from src.factor_graph.predictive_covariance import (
    reliability_to_sigma,
    combine_prediction_with_measurement_quality,
)
from src.odometry.relative_factor_data import (
    RelativeFactorData,
)


EARTH_RADIUS_M = 6378137.0


@dataclass
class FourSensorMeasurements:
    gps_local: np.ndarray
    imu_gyro: np.ndarray
    timestamps: list
    lidar_between: np.ndarray
    lidar_valid: np.ndarray
    lidar_quality: np.ndarray
    camera_between: np.ndarray
    camera_valid: np.ndarray
    camera_quality: np.ndarray


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
        "timestamp"
    ):
        values = np.asarray(
            [
                float(
                    t.timestamp()
                )
                for t in timestamps
            ],
            dtype=np.float64,
        )

    else:
        values = np.asarray(
            timestamps,
            dtype=np.float64,
        )

    return values


def gps_geodetic_to_local(
    latitude_longitude_altitude,
):
    """
    Convert KITTI OXTS [lat, lon, alt] to a local tangent approximation
    in meters relative to the first frame.

    This is adequate for one KITTI raw drive and avoids introducing an
    external geodesy dependency.
    """
    lla = np.asarray(
        latitude_longitude_altitude,
        dtype=np.float64,
    )

    if (
        lla.ndim != 2
        or lla.shape[1] != 3
    ):
        raise ValueError(
            "GPS input must have shape (N, 3) = [lat, lon, alt]"
        )

    lat0 = np.deg2rad(
        lla[
            0,
            0
        ]
    )

    lon0 = np.deg2rad(
        lla[
            0,
            1
        ]
    )

    alt0 = float(
        lla[
            0,
            2
        ]
    )

    lat = np.deg2rad(
        lla[
            :,
            0
        ]
    )

    lon = np.deg2rad(
        lla[
            :,
            1
        ]
    )

    x = (
        EARTH_RADIUS_M
        *
        np.cos(
            lat0
        )
        *
        (
            lon
            -
            lon0
        )
    )

    y = (
        EARTH_RADIUS_M
        *
        (
            lat
            -
            lat0
        )
    )

    z = (
        lla[
            :,
            2
        ]
        -
        alt0
    )

    return np.column_stack(
        [
            x,
            y,
            z,
        ]
    )


def load_four_sensor_measurements(
    sequence_path,
    body_relative_dir,
):
    from src.loader.gps_loader import GPSLoader
    from src.loader.imu_loader import IMULoader

    sequence_path = Path(
        sequence_path
    )

    body_relative_dir = Path(
        body_relative_dir
    )

    gps_loader = GPSLoader(
        sequence_path
    )

    imu_loader = IMULoader(
        sequence_path
    )

    n_sensor = min(
        len(
            gps_loader
        ),
        len(
            imu_loader
        ),
    )

    gps_lla = np.asarray(
        [
            gps_loader[
                i
            ][
                "position"
            ]
            for i in range(
                n_sensor
            )
        ],
        dtype=np.float64,
    )

    gps_local = gps_geodetic_to_local(
        gps_lla
    )

    imu_gyro = np.asarray(
        [
            imu_loader[
                i
            ][
                "angular_velocity"
            ]
            for i in range(
                n_sensor
            )
        ],
        dtype=np.float64,
    )

    timestamps = [
        imu_loader[
            i
        ][
            "timestamp"
        ]
        for i in range(
            n_sensor
        )
    ]

    lidar = RelativeFactorData.load(
        body_relative_dir
        /
        "lidar_factor_data.npz"
    )

    camera = RelativeFactorData.load(
        body_relative_dir
        /
        "camera_factor_data.npz"
    )

    n = min(
        n_sensor,
        len(
            lidar
        )
        +
        1,
        len(
            camera
        )
        +
        1,
    )

    return FourSensorMeasurements(
        gps_local=
            gps_local[
                :n
            ],
        imu_gyro=
            imu_gyro[
                :n
            ],
        timestamps=
            timestamps[
                :n
            ],
        lidar_between=
            lidar.between_measurements[
                :n - 1
            ],
        lidar_valid=
            lidar.valid[
                :n - 1
            ],
        lidar_quality=
            lidar.quality[
                :n - 1
            ],
        camera_between=
            camera.between_measurements[
                :n - 1
            ],
        camera_valid=
            camera.valid[
                :n - 1
            ],
        camera_quality=
            camera.quality[
                :n - 1
            ],
    )


def _load_reliability(
    path,
    n,
):
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
            f"Reliability file too short: {path}"
        )

    return np.clip(
        values[
            :n
        ],
        0.0,
        1.0,
    )


def load_predictive_reliabilities(
    prediction_dir,
    n,
):
    prediction_dir = Path(
        prediction_dir
    )

    result = {}

    for sensor in (
        "gps",
        "imu",
        "lidar",
        "camera",
    ):
        result[
            sensor
        ] = _load_reliability(
            prediction_dir
            /
            f"{sensor}_predictive_prior_target_aligned.txt",
            n,
        )

    return result


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


def _imu_between_pose(
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


def _initial_trajectory_from_lidar(
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
        if measurements.lidar_valid[
            i
        ]:
            delta = _pose3_from_matrix(
                gtsam,
                measurements.lidar_between[
                    i
                ],
            )
        elif measurements.camera_valid[
            i
        ]:
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

    # Translation offset to GPS first frame is zero by local definition.
    return poses


def run_four_sensor_factor_graph(
    measurements,
    output_dir,
    mode="fixed",
    prediction_dir=None,
    use_gps=True,
    use_imu=True,
    use_lidar=True,
    use_camera=True,
    config=None,
):
    """
    Build and optimize a Pose3 factor graph.

    mode:
        "fixed"
        "predictive"
    """
    try:
        import gtsam
    except Exception as exc:
        raise ImportError(
            "gtsam is required for four-sensor factor graph"
        ) from exc

    if config is None:
        config = FourSensorFactorConfig()

    if mode not in (
        "fixed",
        "predictive",
    ):
        raise ValueError(
            "mode must be 'fixed' or 'predictive'"
        )

    n = len(
        measurements.gps_local
    )

    if n < 2:
        raise ValueError(
            "At least 2 frames are required"
        )

    reliabilities = None

    if mode == "predictive":
        if prediction_dir is None:
            raise ValueError(
                "prediction_dir is required in predictive mode"
            )

        reliabilities = load_predictive_reliabilities(
            prediction_dir,
            n,
        )

    graph = gtsam.NonlinearFactorGraph()

    initial = gtsam.Values()

    poses0 = _initial_trajectory_from_lidar(
        gtsam,
        measurements,
    )

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

    factor_counts = {
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
        if use_gps:
            if mode == "fixed":
                sigma_gps = (
                    config.gps_fixed_sigma
                )
            else:
                sigma_gps = float(
                    reliability_to_sigma(
                        reliabilities[
                            "gps"
                        ][i],
                        config.gps_sigma_min,
                        config.gps_sigma_max,
                        config.gps_gamma,
                    )
                )

            gps_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
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

            factor_counts[
                "gps"
            ] += 1

        if i >= n - 1:
            continue

        if len(
            times
        ) == n:
            dt = float(
                times[
                    i + 1
                ]
                -
                times[
                    i
                ]
            )
        else:
            dt = 0.1

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

        if use_imu:
            if mode == "fixed":
                imu_rot_sigma = (
                    config.imu_rotation_sigma
                )
            else:
                imu_rot_sigma = float(
                    reliability_to_sigma(
                        reliabilities[
                            "imu"
                        ][
                            i + 1
                        ],
                        config.imu_rot_sigma_min,
                        config.imu_rot_sigma_max,
                        config.imu_gamma,
                    )
                )

            imu_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.asarray(
                        [
                            imu_rot_sigma,
                            imu_rot_sigma,
                            imu_rot_sigma,
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

            factor_counts[
                "imu"
            ] += 1

        if (
            use_lidar
            and
            measurements.lidar_valid[
                i
            ]
            and
            measurements.lidar_quality[
                i
            ]
            >=
            config.lidar_quality_min
        ):
            if mode == "fixed":
                lidar_rot_sigma = (
                    config.lidar_rotation_sigma
                )

                lidar_trans_sigma = (
                    config.lidar_translation_sigma
                )

            else:
                effective = float(
                    combine_prediction_with_measurement_quality(
                        reliabilities[
                            "lidar"
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

            factor_counts[
                "lidar"
            ] += 1

        if (
            use_camera
            and
            measurements.camera_valid[
                i
            ]
            and
            measurements.camera_quality[
                i
            ]
            >=
            config.camera_quality_min
        ):
            if mode == "fixed":
                camera_rot_sigma = (
                    config.camera_rotation_sigma
                )

                camera_trans_sigma = (
                    config.camera_translation_sigma
                )

            else:
                effective = float(
                    combine_prediction_with_measurement_quality(
                        reliabilities[
                            "camera"
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

            camera_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
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

            factor_counts[
                "camera"
            ] += 1

        if i % 500 == 0:
            print(
                "Add factors:",
                i,
            )

    print(
        "Graph factors:",
        graph.size()
    )

    print(
        "Factor counts:",
        factor_counts,
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

        poses[
            i
        ] = _pose3_to_matrix(
            pose
        )

        trajectory[
            i
        ] = poses[
            i,
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

    np.savetxt(
        output_dir
        /
        "gps_local_reference.txt",
        measurements.gps_local[
            :n
        ],
        fmt="%.8f",
    )

    with (
        output_dir
        /
        "factor_counts.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        for key, value in factor_counts.items():
            file.write(
                f"{key}={value}\n"
            )

    return trajectory, poses, factor_counts
