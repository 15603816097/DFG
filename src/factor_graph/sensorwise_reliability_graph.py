from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import numpy as np

from src.factor_graph.predictive_covariance import reliability_to_sigma
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


SENSORS = ("gps", "imu", "lidar", "camera")


@dataclass(frozen=True)
class ReliabilitySource:
    """
    reliability mode for one sensor:

        fixed       -> use the original fixed covariance
        predictive  -> use V1 predicted factor reliability
        oracle      -> use oracle factor reliability
    """
    mode: str = "fixed"

    def __post_init__(self):
        if self.mode not in ("fixed", "predictive", "oracle"):
            raise ValueError(
                "mode must be one of: fixed, predictive, oracle"
            )


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
    omega = np.asarray(
        gyro_t,
        dtype=np.float64,
    )

    rotvec = omega * float(dt)

    R = gtsam.Rot3.Expmap(rotvec)

    return gtsam.Pose3(
        R,
        gtsam.Point3(
            0.0,
            0.0,
            0.0,
        ),
    )


def _initial_trajectory(gtsam, measurements):
    """
    Initialization only.

    Prefer LiDAR relative odometry, then camera.
    This does not mean those sensors are active factors in every ablation.
    Keeping the initialization identical across cases is important for fairness.
    """
    n = len(measurements.gps_local)

    poses = [
        gtsam.Pose3()
    ]

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


def _load_prediction_file(
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
        raise FileNotFoundError(path)

    values = np.loadtxt(
        path,
        dtype=np.float64,
    ).reshape(-1)

    if len(values) < n:
        raise ValueError(
            f"{path} has {len(values)} values, expected at least {n}"
        )

    return np.clip(
        values[:n],
        0.0,
        1.0,
    )


def _load_oracle_file(
    oracle_npz,
    sensor,
    n,
):
    key = f"{sensor}_reliability"

    if key not in oracle_npz.files:
        raise KeyError(
            f"Missing {key} in oracle file. "
            f"Available: {oracle_npz.files}"
        )

    values = np.asarray(
        oracle_npz[key],
        dtype=np.float64,
    ).reshape(-1)

    if len(values) < n:
        raise ValueError(
            f"Oracle {sensor} length {len(values)} < {n}"
        )

    return np.clip(
        values[:n],
        0.0,
        1.0,
    )


def _prepare_reliability(
    n,
    sources: Dict[str, ReliabilitySource],
    predictive_dir=None,
    oracle_path=None,
):
    result = {
        sensor: None
        for sensor in SENSORS
    }

    oracle_npz = None

    if any(
        sources[sensor].mode == "oracle"
        for sensor in SENSORS
    ):
        if oracle_path is None:
            raise ValueError(
                "oracle_path is required for oracle ablation"
            )

        oracle_npz = np.load(
            oracle_path,
            allow_pickle=False,
        )

    for sensor in SENSORS:
        mode = sources[sensor].mode

        if mode == "predictive":
            if predictive_dir is None:
                raise ValueError(
                    "predictive_dir is required for predictive ablation"
                )

            result[sensor] = _load_prediction_file(
                predictive_dir,
                sensor,
                n,
            )

        elif mode == "oracle":
            result[sensor] = _load_oracle_file(
                oracle_npz,
                sensor,
                n,
            )

    return result


def _gps_sigma(
    config,
    mode,
    reliability,
):
    if mode == "fixed":
        return float(
            config.gps_fixed_sigma
        )

    return float(
        reliability_to_sigma(
            reliability,
            config.gps_sigma_min,
            config.gps_sigma_max,
            config.gps_gamma,
        )
    )


def _imu_sigma(
    config,
    mode,
    reliability,
):
    if mode == "fixed":
        return float(
            config.imu_rotation_sigma
        )

    return float(
        reliability_to_sigma(
            reliability,
            config.imu_rot_sigma_min,
            config.imu_rot_sigma_max,
            config.imu_gamma,
        )
    )


def _lidar_sigmas(
    config,
    mode,
    reliability,
):
    if mode == "fixed":
        return (
            float(
                config.lidar_rotation_sigma
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
        trans,
    )


def _camera_sigmas(
    config,
    mode,
    reliability,
):
    if mode == "fixed":
        return (
            float(
                config.camera_rotation_sigma
            ),
            float(
                config.camera_translation_sigma
            ),
        )

    rot = float(
        reliability_to_sigma(
            reliability,
            config.camera_rot_sigma_min,
            config.camera_rot_sigma_max,
            config.camera_gamma,
        )
    )

    trans = float(
        reliability_to_sigma(
            reliability,
            config.camera_trans_sigma_min,
            config.camera_trans_sigma_max,
            config.camera_gamma,
        )
    )

    return (
        rot,
        trans,
    )


def run_sensorwise_reliability_graph(
    measurements,
    output_dir,
    sources: Dict[str, ReliabilitySource],
    predictive_dir=None,
    oracle_path=None,
    config=None,
):
    """
    Fair sensor-wise reliability ablation.

    SAME:
        degraded measurements
        graph topology
        initialization
        optimizer
        covariance mapping

    ONLY difference:
        which sensor is fixed / predictive / oracle.
    """
    try:
        import gtsam
    except Exception as exc:
        raise ImportError(
            "gtsam is required"
        ) from exc

    if config is None:
        config = FourSensorFactorConfig()

    for sensor in SENSORS:
        if sensor not in sources:
            sources[sensor] = ReliabilitySource("fixed")

    n = len(
        measurements.gps_local
    )

    if n < 2:
        raise ValueError(
            "Need at least 2 frames"
        )

    reliability = _prepare_reliability(
        n,
        sources,
        predictive_dir=
            predictive_dir,
        oracle_path=
            oracle_path,
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
        initial.insert(
            i,
            poses0[i],
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

    sigma_stats = {
        sensor:
            []
        for sensor in SENSORS
    }

    for i in range(n):
        # ----------------------------------------------------------
        # GPS
        # ----------------------------------------------------------
        gps_mode = sources["gps"].mode

        gps_r = (
            1.0
            if gps_mode == "fixed"
            else reliability["gps"][i]
        )

        gps_sigma = _gps_sigma(
            config,
            gps_mode,
            gps_r,
        )

        sigma_stats["gps"].append(
            gps_sigma
        )

        gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
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

        gps_pose = gtsam.Pose3(
            gtsam.Rot3(),
            gtsam.Point3(
                float(
                    measurements.gps_local[i, 0]
                ),
                float(
                    measurements.gps_local[i, 1]
                ),
                float(
                    measurements.gps_local[i, 2]
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

        counts["gps"] += 1

        if i >= n - 1:
            continue

        dt = (
            float(
                times[i + 1]
                -
                times[i]
            )
            if len(times) == n
            else
            0.1
        )

        if (
            not np.isfinite(dt)
            or dt <= 0.0
            or dt > 1.0
        ):
            dt = 0.1

        # ----------------------------------------------------------
        # IMU
        # ----------------------------------------------------------
        imu_mode = sources["imu"].mode

        imu_r = (
            1.0
            if imu_mode == "fixed"
            else reliability["imu"][i + 1]
        )

        imu_sigma = _imu_sigma(
            config,
            imu_mode,
            imu_r,
        )

        sigma_stats["imu"].append(
            imu_sigma
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

        # ----------------------------------------------------------
        # LiDAR
        # ----------------------------------------------------------
        if (
            bool(
                measurements.lidar_valid[i]
            )
            and
            float(
                measurements.lidar_quality[i]
            )
            >=
            config.lidar_quality_min
        ):
            lidar_mode = sources["lidar"].mode

            lidar_r = (
                1.0
                if lidar_mode == "fixed"
                else reliability["lidar"][i + 1]
            )

            lidar_rot_sigma, lidar_trans_sigma = _lidar_sigmas(
                config,
                lidar_mode,
                lidar_r,
            )

            sigma_stats["lidar"].append(
                lidar_trans_sigma
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
                    _pose3_from_matrix(
                        gtsam,
                        measurements.lidar_between[i],
                    ),
                    lidar_noise,
                )
            )

            counts["lidar"] += 1

        # ----------------------------------------------------------
        # Camera
        # ----------------------------------------------------------
        if (
            bool(
                measurements.camera_valid[i]
            )
            and
            float(
                measurements.camera_quality[i]
            )
            >=
            config.camera_quality_min
        ):
            camera_mode = sources["camera"].mode

            camera_r = (
                1.0
                if camera_mode == "fixed"
                else reliability["camera"][i + 1]
            )

            camera_rot_sigma, camera_trans_sigma = _camera_sigmas(
                config,
                camera_mode,
                camera_r,
            )

            sigma_stats["camera"].append(
                camera_trans_sigma
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
                    _pose3_from_matrix(
                        gtsam,
                        measurements.camera_between[i],
                    ),
                    camera_noise,
                )
            )

            counts["camera"] += 1

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
        "Modes:",
        {
            sensor:
                sources[sensor].mode
            for sensor in SENSORS
        },
    )

    for sensor in SENSORS:
        values = np.asarray(
            sigma_stats[sensor],
            dtype=np.float64,
        )

        if len(values) > 0:
            print(
                f"{sensor:8s} sigma "
                f"min/max/mean: "
                f"{values.min():.6f} / "
                f"{values.max():.6f} / "
                f"{values.mean():.6f}"
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

    for i in range(n):
        pose = result.atPose3(
            i
        )

        T = _pose3_to_matrix(
            pose
        )

        poses[i] = T

        trajectory[i] = T[:3, 3]

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
        "modes.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        for sensor in SENSORS:
            file.write(
                f"{sensor}={sources[sensor].mode}\n"
            )

    return (
        trajectory,
        poses,
        counts,
    )
