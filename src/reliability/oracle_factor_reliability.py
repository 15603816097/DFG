from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OracleReliabilityConfig:
    """
    Oracle factor-reliability mapping.

    These values are intentionally interpretable and conservative.
    They are not the final learned model parameters.

    Reliability is mapped from actual factor measurement error with:

        r = exp(-(e / scale)^2)

    For relative-pose factors, translation and rotation reliabilities are
    multiplied so both geometric components matter.
    """

    gps_position_scale_m: float = 5.0

    imu_rotation_scale_deg: float = 2.0

    lidar_translation_scale_m: float = 0.50
    lidar_rotation_scale_deg: float = 1.0

    camera_translation_scale_m: float = 0.75
    camera_rotation_scale_deg: float = 1.5

    minimum_reliability: float = 0.01
    maximum_reliability: float = 1.00


def project_rotation(R):
    R = np.asarray(
        R,
        dtype=np.float64,
    )

    U, _, Vt = np.linalg.svd(
        R
    )

    result = (
        U
        @
        Vt
    )

    if np.linalg.det(
        result
    ) < 0.0:
        U[:, -1] *= -1.0

        result = (
            U
            @
            Vt
        )

    return result


def rotation_angle_rad(R):
    R = project_rotation(
        R
    )

    value = (
        np.trace(
            R
        )
        -
        1.0
    ) * 0.5

    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return float(
        np.arccos(
            value
        )
    )


def invert_transform(T):
    T = np.asarray(
        T,
        dtype=np.float64,
    )

    result = np.eye(
        4,
        dtype=np.float64,
    )

    R = project_rotation(
        T[
            :3,
            :3
        ]
    )

    t = T[
        :3,
        3
    ]

    result[
        :3,
        :3
    ] = R.T

    result[
        :3,
        3
    ] = (
        -
        R.T
        @
        t
    )

    return result


def relative_transform(
    T_world_from_a,
    T_world_from_b,
):
    """
    Return pose increment Z_ab:

        T_world_from_b
        =
        T_world_from_a
        @
        Z_ab
    """
    return (
        invert_transform(
            T_world_from_a
        )
        @
        T_world_from_b
    )


def relative_pose_error(
    measured_between,
    reference_between,
):
    measured_between = np.asarray(
        measured_between,
        dtype=np.float64,
    )

    reference_between = np.asarray(
        reference_between,
        dtype=np.float64,
    )

    translation_error = float(
        np.linalg.norm(
            measured_between[
                :3,
                3
            ]
            -
            reference_between[
                :3,
                3
            ]
        )
    )

    R_error = (
        measured_between[
            :3,
            :3
        ].T
        @
        reference_between[
            :3,
            :3
        ]
    )

    rotation_error_deg = float(
        np.degrees(
            rotation_angle_rad(
                R_error
            )
        )
    )

    return (
        translation_error,
        rotation_error_deg,
    )


def error_to_reliability(
    error,
    scale,
    minimum=0.01,
    maximum=1.0,
):
    error = np.asarray(
        error,
        dtype=np.float64,
    )

    scale = max(
        float(
            scale
        ),
        1e-12,
    )

    reliability = np.exp(
        -
        (
            error
            /
            scale
        )
        ** 2
    )

    return np.clip(
        reliability,
        float(
            minimum
        ),
        float(
            maximum
        ),
    )


def _combine_pose_reliability(
    translation_error,
    rotation_error_deg,
    translation_scale,
    rotation_scale_deg,
    config,
):
    r_t = error_to_reliability(
        translation_error,
        translation_scale,
        config.minimum_reliability,
        config.maximum_reliability,
    )

    r_r = error_to_reliability(
        rotation_error_deg,
        rotation_scale_deg,
        config.minimum_reliability,
        config.maximum_reliability,
    )

    return np.clip(
        r_t
        *
        r_r,
        config.minimum_reliability,
        config.maximum_reliability,
    )


def compute_oracle_factor_reliability(
    clean_gps_local,
    degraded_gps_local,
    degraded_imu_gyro,
    timestamps_seconds,
    lidar_between,
    lidar_valid,
    camera_between,
    camera_valid,
    reference_poses,
    config=None,
):
    """
    Build four oracle reliabilities using actual factor error.

    reference_poses
    ---------------
    Shape: (N, 4, 4)

    These are body poses in a common world frame.

    GPS oracle:
        position error vs clean local GPS reference.

    IMU oracle:
        gyro-integrated rotation increment vs reference relative rotation.

    LiDAR / Camera oracle:
        measured relative pose vs reference relative pose.

    Output reliabilities are aligned to target frame j:

        reliability[j] is the reliability of the factor that ends at j.

    Frame 0 is assigned reliability 1.0 because no previous factor exists.
    """
    if config is None:
        config = OracleReliabilityConfig()

    clean_gps_local = np.asarray(
        clean_gps_local,
        dtype=np.float64,
    )

    degraded_gps_local = np.asarray(
        degraded_gps_local,
        dtype=np.float64,
    )

    degraded_imu_gyro = np.asarray(
        degraded_imu_gyro,
        dtype=np.float64,
    )

    timestamps_seconds = np.asarray(
        timestamps_seconds,
        dtype=np.float64,
    ).reshape(-1)

    reference_poses = np.asarray(
        reference_poses,
        dtype=np.float64,
    )

    n = min(
        len(
            clean_gps_local
        ),
        len(
            degraded_gps_local
        ),
        len(
            degraded_imu_gyro
        ),
        len(
            timestamps_seconds
        ),
        len(
            reference_poses
        ),
        len(
            lidar_between
        )
        +
        1,
        len(
            camera_between
        )
        +
        1,
    )

    clean_gps_local = clean_gps_local[
        :n
    ]

    degraded_gps_local = degraded_gps_local[
        :n
    ]

    degraded_imu_gyro = degraded_imu_gyro[
        :n
    ]

    timestamps_seconds = timestamps_seconds[
        :n
    ]

    reference_poses = reference_poses[
        :n
    ]

    gps_error = np.linalg.norm(
        degraded_gps_local
        -
        clean_gps_local,
        axis=1,
    )

    gps_reliability = error_to_reliability(
        gps_error,
        config.gps_position_scale_m,
        config.minimum_reliability,
        config.maximum_reliability,
    )

    imu_error_deg = np.zeros(
        n,
        dtype=np.float64,
    )

    imu_reliability = np.ones(
        n,
        dtype=np.float64,
    )

    lidar_translation_error = np.zeros(
        n,
        dtype=np.float64,
    )

    lidar_rotation_error_deg = np.zeros(
        n,
        dtype=np.float64,
    )

    lidar_reliability = np.ones(
        n,
        dtype=np.float64,
    )

    camera_translation_error = np.zeros(
        n,
        dtype=np.float64,
    )

    camera_rotation_error_deg = np.zeros(
        n,
        dtype=np.float64,
    )

    camera_reliability = np.ones(
        n,
        dtype=np.float64,
    )

    for i in range(
        n
        -
        1
    ):
        dt = float(
            timestamps_seconds[
                i + 1
            ]
            -
            timestamps_seconds[
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

        reference_between = relative_transform(
            reference_poses[
                i
            ],
            reference_poses[
                i + 1
            ],
        )

        # ----------------------------------------------------------
        # IMU rotation oracle
        # ----------------------------------------------------------
        omega = degraded_imu_gyro[
            i
        ]

        angle = np.linalg.norm(
            omega
            *
            dt
        )

        if angle <= 1e-12:
            R_imu = np.eye(
                3,
                dtype=np.float64,
            )

        else:
            axis = (
                omega
                /
                max(
                    np.linalg.norm(
                        omega
                    ),
                    1e-12,
                )
            )

            theta = float(
                angle
            )

            K = np.asarray(
                [
                    [
                        0.0,
                        -axis[2],
                        axis[1],
                    ],
                    [
                        axis[2],
                        0.0,
                        -axis[0],
                    ],
                    [
                        -axis[1],
                        axis[0],
                        0.0,
                    ],
                ],
                dtype=np.float64,
            )

            R_imu = (
                np.eye(
                    3
                )
                +
                np.sin(
                    theta
                )
                *
                K
                +
                (
                    1.0
                    -
                    np.cos(
                        theta
                    )
                )
                *
                (
                    K
                    @
                    K
                )
            )

        R_error = (
            R_imu.T
            @
            reference_between[
                :3,
                :3
            ]
        )

        imu_error_deg[
            i + 1
        ] = np.degrees(
            rotation_angle_rad(
                R_error
            )
        )

        imu_reliability[
            i + 1
        ] = error_to_reliability(
            imu_error_deg[
                i + 1
            ],
            config.imu_rotation_scale_deg,
            config.minimum_reliability,
            config.maximum_reliability,
        )

        # ----------------------------------------------------------
        # LiDAR factor oracle
        # ----------------------------------------------------------
        if bool(
            lidar_valid[
                i
            ]
        ):
            t_error, r_error = relative_pose_error(
                lidar_between[
                    i
                ],
                reference_between,
            )

            lidar_translation_error[
                i + 1
            ] = t_error

            lidar_rotation_error_deg[
                i + 1
            ] = r_error

            lidar_reliability[
                i + 1
            ] = _combine_pose_reliability(
                t_error,
                r_error,
                config.lidar_translation_scale_m,
                config.lidar_rotation_scale_deg,
                config,
            )

        else:
            lidar_translation_error[
                i + 1
            ] = np.inf

            lidar_rotation_error_deg[
                i + 1
            ] = np.inf

            lidar_reliability[
                i + 1
            ] = config.minimum_reliability

        # ----------------------------------------------------------
        # Camera factor oracle
        # ----------------------------------------------------------
        if bool(
            camera_valid[
                i
            ]
        ):
            t_error, r_error = relative_pose_error(
                camera_between[
                    i
                ],
                reference_between,
            )

            camera_translation_error[
                i + 1
            ] = t_error

            camera_rotation_error_deg[
                i + 1
            ] = r_error

            camera_reliability[
                i + 1
            ] = _combine_pose_reliability(
                t_error,
                r_error,
                config.camera_translation_scale_m,
                config.camera_rotation_scale_deg,
                config,
            )

        else:
            camera_translation_error[
                i + 1
            ] = np.inf

            camera_rotation_error_deg[
                i + 1
            ] = np.inf

            camera_reliability[
                i + 1
            ] = config.minimum_reliability

    return {
        "gps_error_m":
            gps_error,

        "gps_reliability":
            gps_reliability,

        "imu_rotation_error_deg":
            imu_error_deg,

        "imu_reliability":
            imu_reliability,

        "lidar_translation_error_m":
            lidar_translation_error,

        "lidar_rotation_error_deg":
            lidar_rotation_error_deg,

        "lidar_reliability":
            lidar_reliability,

        "camera_translation_error_m":
            camera_translation_error,

        "camera_rotation_error_deg":
            camera_rotation_error_deg,

        "camera_reliability":
            camera_reliability,
    }
