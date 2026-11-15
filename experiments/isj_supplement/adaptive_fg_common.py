from __future__ import annotations
"""Shared Stage-7 graph builder for diagnostic and nonlearned adaptive baselines.

Matches the official Stage-7 GPS/IMU/LiDAR/Camera factors and initial prior.
Adaptation (when requested) uses ONLY residuals from the saved fixed-FGO
Stage-1 trajectory. No truth trajectory or model predictions are used.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence import run_stage7_factor_graph as s7
from experiments.cross_sequence.stage7_config import (
    GPS_FIXED, IMU_R0, CAM_R0, CAM_T0, LIDAR_R0, LIDAR_T0, sequence_out
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig

# Square roots of 0.95 quantiles of chi-square with 3 and 6 degrees of freedom.
# All constants declared before running new supplemental baselines.
CHI_3D_95 = 2.795
CHI_6D_95 = 3.548
MAX_SIGMA_MULTIPLIER = 4.0


def load_first_stage(seq: str):
    m = s7.load_measurements(seq)
    n = len(m.gps_local)
    path = sequence_out(seq) / "factor_graph" / "FINAL_S0_fixed" / "poses.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    poses = np.load(path, allow_pickle=False)
    validate_poses(poses, n)
    return m, poses


def validate_poses(poses: np.ndarray, n: int) -> None:
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"Expected finite ({n},4,4) poses, got {poses.shape}")


def sigma_multiplier(normalized_residual: np.ndarray, cutoff: float) -> float:
    """Huber-like residual covariance inflation, bounded without GT tuning.

    With q = ||normalized 3-D or 6-D residual|| and cutoff c:
      sigma/sigma0 = sqrt(max(1, q/c)), capped at four.
    This is one frozen outer-loop IRLS-style update, not a learned estimator.
    """
    q = float(np.linalg.norm(np.asarray(normalized_residual, dtype=float)))
    if not np.isfinite(q) or cutoff <= 0:
        raise ValueError("Invalid residual or cutoff")
    return float(np.clip(np.sqrt(max(1.0, q / cutoff)), 1.0,
                         MAX_SIGMA_MULTIPLIER))


def build_factor_graph(m, initial_poses: np.ndarray, adaptive_sensors=()):
    """Build Stage-7 equivalent graph, optionally inflate GPS/IMU/camera noise.

    For the adaptive variant, FIRST-stage frozen poses supply both residuals
    and second-stage initialization. Fixed LiDAR noise is always preserved.
    """
    import gtsam

    n = len(m.gps_local)
    validate_poses(initial_poses, n)
    selected = frozenset(adaptive_sensors)
    unknown = selected - {"gps", "imu", "camera"}
    if unknown:
        raise ValueError(f"Unknown adaptation sensors: {sorted(unknown)}")
    cfg = FourSensorFactorConfig()
    gt_poses = [s7._pose3_from_matrix(gtsam, p) for p in initial_poses]
    first_pose = s7.initial_pose_from_oxts(gtsam, m)
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    stats = {"gps": [], "imu": [], "camera": [],
             "factor_counts": {"gps": 0, "imu": 0, "camera": 0, "lidar": 0}}

    prior_sigmas = np.asarray([cfg.prior_rotation_sigma] * 3
                              + [cfg.prior_translation_sigma] * 3, float)
    graph.add(gtsam.PriorFactorPose3(
        0, first_pose, gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
    ))

    times = np.asarray(m.timestamps, dtype=float)
    for i, pose in enumerate(gt_poses):
        initial.insert(i, pose)

    for i in range(n):
        gps_measurement = gtsam.Pose3(
            gtsam.Rot3(), gtsam.Point3(*map(float, m.gps_local[i, :3])))
        gps_weight = 1.0
        if "gps" in selected:
            r_gps = (np.asarray(gt_poses[i].translation(), float)
                     - np.asarray(m.gps_local[i, :3], float)) / GPS_FIXED
            gps_weight = sigma_multiplier(r_gps, CHI_3D_95)
        stats["gps"].append(gps_weight)
        graph.add(gtsam.PriorFactorPose3(
            i, gps_measurement,
            gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray([1e6] * 3 + [GPS_FIXED * gps_weight] * 3, float))
        ))
        stats["factor_counts"]["gps"] += 1

        if i == n - 1:
            continue
        dt = float(times[i + 1] - times[i])
        if not np.isfinite(dt) or dt <= 0 or dt > 1:
            dt = 0.1
        estimate_delta = gt_poses[i].between(gt_poses[i + 1])
        imu_measurement = s7._imu_between_pose(gtsam, m.imu_gyro[i], dt)
        imu_weight = 1.0
        if "imu" in selected:
            r_imu = np.asarray(imu_measurement.localCoordinates(estimate_delta),
                               dtype=float)[:3] / IMU_R0
            imu_weight = sigma_multiplier(r_imu, CHI_3D_95)
        stats["imu"].append(imu_weight)
        graph.add(gtsam.BetweenFactorPose3(
            i, i + 1, imu_measurement,
            gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray([IMU_R0 * imu_weight] * 3
                           + [cfg.imu_translation_sigma] * 3, float))
        ))
        stats["factor_counts"]["imu"] += 1

        if bool(m.lidar_valid[i]):
            lidar_measurement = s7._pose3_from_matrix(gtsam, m.lidar_between[i])
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1, lidar_measurement,
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.asarray([LIDAR_R0] * 3 + [LIDAR_T0] * 3, float))
            ))
            stats["factor_counts"]["lidar"] += 1

        if bool(m.camera_valid[i]):
            camera_measurement = s7._pose3_from_matrix(
                gtsam, m.camera_between[i])
            camera_weight = 1.0
            if "camera" in selected:
                e = np.asarray(camera_measurement.localCoordinates(estimate_delta),
                               dtype=float)
                if e.shape != (6,):
                    raise ValueError(f"Camera pose residual shape {e.shape}")
                scaled = e / np.asarray(
                    [CAM_R0] * 3 + [CAM_T0] * 3, dtype=float)
                camera_weight = sigma_multiplier(scaled, CHI_6D_95)
            stats["camera"].append(camera_weight)
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1, camera_measurement,
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.asarray([CAM_R0 * camera_weight] * 3
                               + [CAM_T0 * camera_weight] * 3, float))
            ))
            stats["factor_counts"]["camera"] += 1

    for sensor in ("gps", "imu", "camera"):
        x = np.asarray(stats[sensor], dtype=float)
        stats[sensor] = {
            "mean_sigma_multiplier": float(x.mean()) if len(x) else None,
            "max_sigma_multiplier": float(x.max()) if len(x) else None,
            "inflated_fraction": float(np.mean(x > 1.000001)) if len(x) else None,
        }
    stats["graph_factor_count"] = int(graph.size())
    return graph, initial, stats


def values_from_poses(poses: np.ndarray):
    import gtsam

    values = gtsam.Values()
    for i, p in enumerate(poses):
        values.insert(i, s7._pose3_from_matrix(gtsam, p))
    return values


def optimize_graph(graph, values):
    import gtsam

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-7)
    return gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()


def export_poses(values, n: int) -> np.ndarray:
    return np.stack([s7._pose3_to_matrix(values.atPose3(i))
                     for i in range(n)], axis=0)
