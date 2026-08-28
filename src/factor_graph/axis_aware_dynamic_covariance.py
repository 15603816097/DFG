from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from src.factor_graph.predictive_covariance import reliability_to_sigma
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


SENSORS = ("gps", "imu", "lidar", "camera")


@dataclass(frozen=True)
class AxisAwarePolicy:
    """
    Axis-aware covariance policy.

    Dynamic means:
        use V1 predictive factor reliability -> existing sensor-specific
        reliability_to_sigma mapping.

    Fixed means:
        preserve the original fixed covariance for that state dimension.

    GTSAM Pose3 diagonal order used here:
        [roll, pitch, yaw, x, y, z]
    """

    name: str

    gps_xy_dynamic: bool = True
    gps_z_dynamic: bool = True

    imu_roll_dynamic: bool = True
    imu_pitch_dynamic: bool = True
    imu_yaw_dynamic: bool = True

    # LiDAR is deliberately protected as fixed in this experiment.
    lidar_dynamic: bool = False

    camera_roll_dynamic: bool = True
    camera_pitch_dynamic: bool = True
    camera_yaw_dynamic: bool = True
    camera_xy_dynamic: bool = True
    camera_z_dynamic: bool = True


POLICIES: Dict[str, AxisAwarePolicy] = {
    # Reproduces the current best learned subset:
    # GPS + IMU + Camera predictive, LiDAR fixed.
    "best_subset_control": AxisAwarePolicy(
        name="best_subset_control",
    ),

    # Only protect GPS altitude.
    "gps_z_fixed": AxisAwarePolicy(
        name="gps_z_fixed",
        gps_z_dynamic=False,
    ),

    # Only protect camera translation Z.
    "camera_z_fixed": AxisAwarePolicy(
        name="camera_z_fixed",
        camera_z_dynamic=False,
    ),

    # Protect both absolute and visual vertical translation.
    "gps_camera_z_fixed": AxisAwarePolicy(
        name="gps_camera_z_fixed",
        gps_z_dynamic=False,
        camera_z_dynamic=False,
    ),

    # Camera keeps only yaw + XY dynamic.
    "camera_rpz_fixed": AxisAwarePolicy(
        name="camera_rpz_fixed",
        camera_roll_dynamic=False,
        camera_pitch_dynamic=False,
        camera_z_dynamic=False,
    ),

    # Full axis-aware proposal:
    # GPS:     XY dynamic, Z fixed
    # IMU:     yaw dynamic, roll/pitch fixed
    # LiDAR:   fully fixed
    # Camera:  yaw + XY dynamic, roll/pitch/Z fixed
    "full_axis_aware": AxisAwarePolicy(
        name="full_axis_aware",
        gps_z_dynamic=False,
        imu_roll_dynamic=False,
        imu_pitch_dynamic=False,
        camera_roll_dynamic=False,
        camera_pitch_dynamic=False,
        camera_z_dynamic=False,
    ),
}


def get_policy(name: str) -> AxisAwarePolicy:
    if name not in POLICIES:
        raise KeyError(
            f"Unknown policy '{name}'. "
            f"Available: {list(POLICIES.keys())}"
        )
    return POLICIES[name]


def _load_prediction(
    prediction_dir,
    sensor,
    n,
):
    path = (
        Path(prediction_dir)
        /
        f"{sensor}_predictive_prior_target_aligned.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing V1 predictive reliability: {path}"
        )

    values = np.loadtxt(
        path,
        dtype=np.float64,
    ).reshape(-1)

    if len(values) < n:
        raise ValueError(
            f"{path}: got {len(values)} values, need {n}"
        )

    return np.clip(
        values[:n],
        0.0,
        1.0,
    )


def load_v1_reliability(
    prediction_dir,
    n,
):
    return {
        sensor:
            _load_prediction(
                prediction_dir,
                sensor,
                n,
            )
        for sensor in SENSORS
    }


def gps_axis_sigmas(
    config,
    reliability,
    policy: AxisAwarePolicy,
):
    dynamic = float(
        reliability_to_sigma(
            reliability,
            config.gps_sigma_min,
            config.gps_sigma_max,
            config.gps_gamma,
        )
    )

    fixed = float(
        config.gps_fixed_sigma
    )

    sigma_x = (
        dynamic
        if policy.gps_xy_dynamic
        else fixed
    )

    sigma_y = (
        dynamic
        if policy.gps_xy_dynamic
        else fixed
    )

    sigma_z = (
        dynamic
        if policy.gps_z_dynamic
        else fixed
    )

    return (
        sigma_x,
        sigma_y,
        sigma_z,
    )


def imu_axis_sigmas(
    config,
    reliability,
    policy: AxisAwarePolicy,
):
    dynamic = float(
        reliability_to_sigma(
            reliability,
            config.imu_rot_sigma_min,
            config.imu_rot_sigma_max,
            config.imu_gamma,
        )
    )

    fixed = float(
        config.imu_rotation_sigma
    )

    sigma_roll = (
        dynamic
        if policy.imu_roll_dynamic
        else fixed
    )

    sigma_pitch = (
        dynamic
        if policy.imu_pitch_dynamic
        else fixed
    )

    sigma_yaw = (
        dynamic
        if policy.imu_yaw_dynamic
        else fixed
    )

    return (
        sigma_roll,
        sigma_pitch,
        sigma_yaw,
    )


def lidar_axis_sigmas(
    config,
    reliability,
    policy: AxisAwarePolicy,
):
    if not policy.lidar_dynamic:
        return (
            float(
                config.lidar_rotation_sigma
            ),
            float(
                config.lidar_rotation_sigma
            ),
            float(
                config.lidar_rotation_sigma
            ),
            float(
                config.lidar_translation_sigma
            ),
            float(
                config.lidar_translation_sigma
            ),
            float(
                config.lidar_translation_sigma
            ),
        )

    rot = float(
        reliability_to_sigma(
            reliability,
            config.lidar_rot_sigma_min,
            config.lidar_rot_sigma_max,
            config.lidar_gamma,
        )
    )

    trans = float(
        reliability_to_sigma(
            reliability,
            config.lidar_trans_sigma_min,
            config.lidar_trans_sigma_max,
            config.lidar_gamma,
        )
    )

    return (
        rot,
        rot,
        rot,
        trans,
        trans,
        trans,
    )


def camera_axis_sigmas(
    config,
    reliability,
    policy: AxisAwarePolicy,
):
    dynamic_rot = float(
        reliability_to_sigma(
            reliability,
            config.camera_rot_sigma_min,
            config.camera_rot_sigma_max,
            config.camera_gamma,
        )
    )

    dynamic_trans = float(
        reliability_to_sigma(
            reliability,
            config.camera_trans_sigma_min,
            config.camera_trans_sigma_max,
            config.camera_gamma,
        )
    )

    fixed_rot = float(
        config.camera_rotation_sigma
    )

    fixed_trans = float(
        config.camera_translation_sigma
    )

    sigma_roll = (
        dynamic_rot
        if policy.camera_roll_dynamic
        else fixed_rot
    )

    sigma_pitch = (
        dynamic_rot
        if policy.camera_pitch_dynamic
        else fixed_rot
    )

    sigma_yaw = (
        dynamic_rot
        if policy.camera_yaw_dynamic
        else fixed_rot
    )

    sigma_x = (
        dynamic_trans
        if policy.camera_xy_dynamic
        else fixed_trans
    )

    sigma_y = (
        dynamic_trans
        if policy.camera_xy_dynamic
        else fixed_trans
    )

    sigma_z = (
        dynamic_trans
        if policy.camera_z_dynamic
        else fixed_trans
    )

    return (
        sigma_roll,
        sigma_pitch,
        sigma_yaw,
        sigma_x,
        sigma_y,
        sigma_z,
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


def _build_initial_values(
    gtsam,
    measurements,
):
    """
    Keep identical initialization across every axis-aware policy.

    LiDAR relative odometry first, camera fallback.
    """
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


def _stat_line(
    name,
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(
        values
    ) == 0:
        return (
            f"{name}: none"
        )

    return (
        f"{name}: "
        f"min={values.min():.6f} "
        f"max={values.max():.6f} "
        f"mean={values.mean():.6f}"
    )


def run_axis_aware_factor_graph(
    measurements,
    output_dir,
    prediction_dir,
    policy_name,
    config=None,
):
    """
    Run the fair axis-aware experiment.

    Fixed across policies:
        degraded measurements
        factor counts
        graph topology
        initialization
        optimizer
        reliability predictor
        scalar reliability-to-sigma mapping

    Changed:
        which Pose3 state dimensions are allowed to use dynamic covariance.
    """
    try:
        import gtsam

    except Exception as exc:
        raise ImportError(
            "gtsam is required"
        ) from exc

    if config is None:
        config = FourSensorFactorConfig()

    policy = get_policy(
        policy_name
    )

    n = len(
        measurements.gps_local
    )

    reliability = load_v1_reliability(
        prediction_dir,
        n,
    )

    graph = gtsam.NonlinearFactorGraph()

    initial = gtsam.Values()

    initial_poses = _build_initial_values(
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
            initial_poses[
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

    diagnostics = {
        "gps_xy":
            [],
        "gps_z":
            [],
        "imu_roll":
            [],
        "imu_pitch":
            [],
        "imu_yaw":
            [],
        "camera_roll":
            [],
        "camera_pitch":
            [],
        "camera_yaw":
            [],
        "camera_xy":
            [],
        "camera_z":
            [],
    }

    for i in range(
        n
    ):
        # ----------------------------------------------------------
        # GPS absolute-position factor
        # ----------------------------------------------------------
        (
            gps_x,
            gps_y,
            gps_z,
        ) = gps_axis_sigmas(
            config,
            reliability[
                "gps"
            ][
                i
            ],
            policy,
        )

        diagnostics[
            "gps_xy"
        ].append(
            0.5
            *
            (
                gps_x
                +
                gps_y
            )
        )

        diagnostics[
            "gps_z"
        ].append(
            gps_z
        )

        gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    1e6,
                    1e6,
                    1e6,
                    gps_x,
                    gps_y,
                    gps_z,
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
        # IMU rotational factor
        # ----------------------------------------------------------
        (
            imu_roll,
            imu_pitch,
            imu_yaw,
        ) = imu_axis_sigmas(
            config,
            reliability[
                "imu"
            ][
                i + 1
            ],
            policy,
        )

        diagnostics[
            "imu_roll"
        ].append(
            imu_roll
        )

        diagnostics[
            "imu_pitch"
        ].append(
            imu_pitch
        )

        diagnostics[
            "imu_yaw"
        ].append(
            imu_yaw
        )

        imu_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    imu_roll,
                    imu_pitch,
                    imu_yaw,
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
        # LiDAR relative factor
        # Fixed in all policies in this experiment.
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
            lidar_sigmas = lidar_axis_sigmas(
                config,
                reliability[
                    "lidar"
                ][
                    i + 1
                ],
                policy,
            )

            lidar_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    lidar_sigmas,
                    dtype=np.float64,
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
        # Camera relative factor
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
            (
                cam_roll,
                cam_pitch,
                cam_yaw,
                cam_x,
                cam_y,
                cam_z,
            ) = camera_axis_sigmas(
                config,
                reliability[
                    "camera"
                ][
                    i + 1
                ],
                policy,
            )

            diagnostics[
                "camera_roll"
            ].append(
                cam_roll
            )

            diagnostics[
                "camera_pitch"
            ].append(
                cam_pitch
            )

            diagnostics[
                "camera_yaw"
            ].append(
                cam_yaw
            )

            diagnostics[
                "camera_xy"
            ].append(
                0.5
                *
                (
                    cam_x
                    +
                    cam_y
                )
            )

            diagnostics[
                "camera_z"
            ].append(
                cam_z
            )

            camera_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    [
                        cam_roll,
                        cam_pitch,
                        cam_yaw,
                        cam_x,
                        cam_y,
                        cam_z,
                    ],
                    dtype=np.float64,
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
        "Policy:",
        policy,
    )

    print(
        _stat_line(
            "GPS XY sigma",
            diagnostics[
                "gps_xy"
            ],
        )
    )

    print(
        _stat_line(
            "GPS Z sigma",
            diagnostics[
                "gps_z"
            ],
        )
    )

    print(
        _stat_line(
            "IMU roll sigma",
            diagnostics[
                "imu_roll"
            ],
        )
    )

    print(
        _stat_line(
            "IMU pitch sigma",
            diagnostics[
                "imu_pitch"
            ],
        )
    )

    print(
        _stat_line(
            "IMU yaw sigma",
            diagnostics[
                "imu_yaw"
            ],
        )
    )

    print(
        _stat_line(
            "Camera roll sigma",
            diagnostics[
                "camera_roll"
            ],
        )
    )

    print(
        _stat_line(
            "Camera pitch sigma",
            diagnostics[
                "camera_pitch"
            ],
        )
    )

    print(
        _stat_line(
            "Camera yaw sigma",
            diagnostics[
                "camera_yaw"
            ],
        )
    )

    print(
        _stat_line(
            "Camera XY sigma",
            diagnostics[
                "camera_xy"
            ],
        )
    )

    print(
        _stat_line(
            "Camera Z sigma",
            diagnostics[
                "camera_z"
            ],
        )
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

    with (
        output_dir
        /
        "policy.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            repr(
                policy
            )
            +
            "\n"
        )

    return (
        trajectory,
        poses,
        counts,
    )
