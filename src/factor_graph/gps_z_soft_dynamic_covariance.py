from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.factor_graph.predictive_covariance import reliability_to_sigma
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


SENSORS = ("gps", "imu", "lidar", "camera")


@dataclass(frozen=True)
class GPSZSoftPolicy:
    alpha_z: float

    def __post_init__(self):
        if not 0.0 <= self.alpha_z <= 1.0:
            raise ValueError(
                f"alpha_z must be in [0, 1], got {self.alpha_z}"
            )


ALPHA_VALUES = (
    0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.00,
)


def alpha_name(alpha: float) -> str:
    return f"alpha_z_{alpha:.2f}".replace(".", "p")


def _timestamp_seconds(timestamps):
    if len(timestamps) == 0:
        return np.asarray([], dtype=np.float64)

    first = timestamps[0]

    if hasattr(first, "timestamp"):
        return np.asarray(
            [float(t.timestamp()) for t in timestamps],
            dtype=np.float64,
        )

    return np.asarray(
        timestamps,
        dtype=np.float64,
    )


def _pose3_from_matrix(gtsam, T):
    T = np.asarray(T, dtype=np.float64)

    return gtsam.Pose3(
        gtsam.Rot3(T[:3, :3]),
        gtsam.Point3(
            float(T[0, 3]),
            float(T[1, 3]),
            float(T[2, 3]),
        ),
    )


def _pose3_to_matrix(pose):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = np.asarray(
        pose.translation(),
        dtype=np.float64,
    ).reshape(3)
    return T


def _imu_between_pose(gtsam, gyro_t, dt):
    gyro_t = np.asarray(
        gyro_t,
        dtype=np.float64,
    )

    rotvec = gyro_t * float(dt)

    return gtsam.Pose3(
        gtsam.Rot3.Expmap(rotvec),
        gtsam.Point3(0.0, 0.0, 0.0),
    )


def _initial_trajectory(gtsam, measurements):
    n = len(measurements.gps_local)

    poses = [gtsam.Pose3()]

    for i in range(n - 1):
        if bool(measurements.lidar_valid[i]):
            delta = _pose3_from_matrix(
                gtsam,
                measurements.lidar_between[i],
            )
        elif bool(measurements.camera_valid[i]):
            delta = _pose3_from_matrix(
                gtsam,
                measurements.camera_between[i],
            )
        else:
            delta = gtsam.Pose3()

        poses.append(
            poses[-1].compose(delta)
        )

    return poses


def _load_prediction(prediction_dir, sensor, n):
    path = (
        Path(prediction_dir)
        / f"{sensor}_predictive_prior_target_aligned.txt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    values = np.loadtxt(
        path,
        dtype=np.float64,
    ).reshape(-1)

    if len(values) < n:
        raise ValueError(
            f"{path}: {len(values)} values, expected >= {n}"
        )

    return np.clip(values[:n], 0.0, 1.0)


def load_v1_reliability(prediction_dir, n):
    return {
        sensor: _load_prediction(
            prediction_dir,
            sensor,
            n,
        )
        for sensor in SENSORS
    }


def gps_dynamic_sigma(config, reliability):
    return float(
        reliability_to_sigma(
            reliability,
            config.gps_sigma_min,
            config.gps_sigma_max,
            config.gps_gamma,
        )
    )


def gps_z_soft_sigma(
    config,
    reliability,
    alpha_z,
):
    """
    sigma_z =
        (1-alpha_z) * sigma_fixed
        + alpha_z * sigma_dynamic
    """
    alpha_z = float(alpha_z)

    dynamic = gps_dynamic_sigma(
        config,
        reliability,
    )

    fixed = float(
        config.gps_fixed_sigma
    )

    return (
        (1.0 - alpha_z) * fixed
        + alpha_z * dynamic
    )


def imu_dynamic_sigma(config, reliability):
    return float(
        reliability_to_sigma(
            reliability,
            config.imu_rot_sigma_min,
            config.imu_rot_sigma_max,
            config.imu_gamma,
        )
    )


def camera_dynamic_sigmas(config, reliability):
    rotation = float(
        reliability_to_sigma(
            reliability,
            config.camera_rot_sigma_min,
            config.camera_rot_sigma_max,
            config.camera_gamma,
        )
    )

    translation = float(
        reliability_to_sigma(
            reliability,
            config.camera_trans_sigma_min,
            config.camera_trans_sigma_max,
            config.camera_gamma,
        )
    )

    return rotation, translation


def run_gps_z_soft_factor_graph(
    measurements,
    output_dir,
    prediction_dir,
    alpha_z,
    config=None,
):
    try:
        import gtsam
    except Exception as exc:
        raise ImportError("gtsam is required") from exc

    policy = GPSZSoftPolicy(
        alpha_z=float(alpha_z)
    )

    if config is None:
        config = FourSensorFactorConfig()

    n = len(measurements.gps_local)

    reliability = load_v1_reliability(
        prediction_dir,
        n,
    )

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    poses0 = _initial_trajectory(
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

    for i in range(n):
        initial.insert(i, poses0[i])

    times = _timestamp_seconds(
        measurements.timestamps
    )

    counts = {
        "gps": 0,
        "imu": 0,
        "lidar": 0,
        "camera": 0,
    }

    gps_xy_sigmas = []
    gps_z_sigmas = []

    for i in range(n):
        gps_r = reliability["gps"][i]

        gps_xy = gps_dynamic_sigma(
            config,
            gps_r,
        )

        gps_z = gps_z_soft_sigma(
            config,
            gps_r,
            policy.alpha_z,
        )

        gps_xy_sigmas.append(gps_xy)
        gps_z_sigmas.append(gps_z)

        gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(
                [
                    1e6,
                    1e6,
                    1e6,
                    gps_xy,
                    gps_xy,
                    gps_z,
                ],
                dtype=np.float64,
            )
        )

        gps_pose = gtsam.Pose3(
            gtsam.Rot3(),
            gtsam.Point3(
                float(measurements.gps_local[i, 0]),
                float(measurements.gps_local[i, 1]),
                float(measurements.gps_local[i, 2]),
            ),
        )

        graph.add(
            gtsam.PriorFactorPose3(
                i,
                gps_pose,
                gps_noise,
            )
        )

        counts["gps"] += 1

        if i >= n - 1:
            continue

        dt = (
            float(times[i + 1] - times[i])
            if len(times) == n
            else 0.1
        )

        if (
            not np.isfinite(dt)
            or dt <= 0.0
            or dt > 1.0
        ):
            dt = 0.1

        # IMU predictive
        imu_r = reliability["imu"][i + 1]

        imu_sigma = imu_dynamic_sigma(
            config,
            imu_r,
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
                _imu_between_pose(
                    gtsam,
                    measurements.imu_gyro[i],
                    dt,
                ),
                imu_noise,
            )
        )

        counts["imu"] += 1

        # LiDAR fixed
        if (
            bool(measurements.lidar_valid[i])
            and
            float(measurements.lidar_quality[i])
            >= config.lidar_quality_min
        ):
            lidar_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    [
                        config.lidar_rotation_sigma,
                        config.lidar_rotation_sigma,
                        config.lidar_rotation_sigma,
                        config.lidar_translation_sigma,
                        config.lidar_translation_sigma,
                        config.lidar_translation_sigma,
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
                        measurements.lidar_between[i],
                    ),
                    lidar_noise,
                )
            )

            counts["lidar"] += 1

        # Camera predictive
        if (
            bool(measurements.camera_valid[i])
            and
            float(measurements.camera_quality[i])
            >= config.camera_quality_min
        ):
            camera_r = reliability[
                "camera"
            ][
                i + 1
            ]

            camera_rot, camera_trans = (
                camera_dynamic_sigmas(
                    config,
                    camera_r,
                )
            )

            camera_noise = gtsam.noiseModel.Diagonal.Sigmas(
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

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    _pose3_from_matrix(
                        gtsam,
                        measurements.camera_between[i],
                    ),
                    camera_noise,
                )
            )

            counts["camera"] += 1

        if i % 500 == 0:
            print("Add factors:", i)

    print("Graph factors:", graph.size())
    print("Factor counts:", counts)
    print("alpha_z:", policy.alpha_z)

    gps_xy_sigmas = np.asarray(
        gps_xy_sigmas,
        dtype=np.float64,
    )

    gps_z_sigmas = np.asarray(
        gps_z_sigmas,
        dtype=np.float64,
    )

    print(
        "GPS XY sigma min/max/mean:",
        float(gps_xy_sigmas.min()),
        float(gps_xy_sigmas.max()),
        float(gps_xy_sigmas.mean()),
    )

    print(
        "GPS Z sigma min/max/mean:",
        float(gps_z_sigmas.min()),
        float(gps_z_sigmas.max()),
        float(gps_z_sigmas.mean()),
    )

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-7)

    print("Optimizing...")

    result = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial,
        params,
    ).optimize()

    trajectory = np.zeros(
        (n, 3),
        dtype=np.float64,
    )

    poses = np.zeros(
        (n, 4, 4),
        dtype=np.float64,
    )

    for i in range(n):
        pose = result.atPose3(i)
        T = _pose3_to_matrix(pose)

        poses[i] = T
        trajectory[i] = T[:3, 3]

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        output_dir / "trajectory.txt",
        trajectory,
        fmt="%.8f",
    )

    np.save(
        output_dir / "poses.npy",
        poses,
    )

    (output_dir / "alpha_z.txt").write_text(
        f"{policy.alpha_z:.10f}\n",
        encoding="utf-8",
    )

    return trajectory, poses, counts
