from __future__ import annotations

"""GPS/IMU versus LiDAR-only local motion consistency (offline, diagnostic).

This module constructs *measurement-only* causal-in-measurement-order
consistency scores for use by a proposed evidence-feedback calibration
module. It does not use clean GPS, evaluation trajectory, oracle labels,
fixed-FGO poses, or any future measurement to form feedback.

GPS: on [t-k, t], compare magnitudes of GPS and chained LiDAR
     displacements. This is rotation-invariant but CAN MISS directional
     errors and constant GPS biases.
IMU: on [t-1,t], compare gyro-integrated SO(3) with LiDAR between rotation.
     NIS scales are the frozen Stage-7 nominal sigmas, which are heuristic,
     NOT calibrated innovation covariance.

LiDAR valid/quality mask is required. q values are a monotone consistency
score, not the calibrated probability that a sensor measurement is valid.
Both GPS and IMU can be wrong when LiDAR is wrong; do not call the signals
statistically independent. Score calculations need only numpy and the
already implemented normalized_innovation_quality.
"""

from dataclasses import dataclass
import math

import numpy as np

from src.reliability.evidence_feedback_calibration import (
    normalized_innovation_quality,
)


@dataclass(frozen=True)
class CrossModalSettings:
    gps_window: int = 5
    gps_sigma_m: float = 5.0
    lidar_translation_sigma_m: float = 0.15
    imu_rotation_sigma_rad: float = 0.03
    lidar_rotation_sigma_rad: float = 0.012857142857142857
    lidar_quality_min: float = 0.10
    gps_chi2_95: float = 3.841458820694124
    imu_chi2_95: float = 7.814727903251179

    def __post_init__(self):
        if not isinstance(self.gps_window, int) or self.gps_window < 1:
            raise ValueError("gps_window must be a positive integer")
        for key in (
            "gps_sigma_m", "lidar_translation_sigma_m",
            "imu_rotation_sigma_rad", "lidar_rotation_sigma_rad",
            "gps_chi2_95", "imu_chi2_95",
        ):
            x = float(getattr(self, key))
            if not np.isfinite(x) or x <= 0.0:
                raise ValueError(f"{key} must be positive finite")
        if not 0.0 <= self.lidar_quality_min <= 1.0:
            raise ValueError("lidar_quality_min must be in [0,1]")


def _rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    """Stable Rodrigues exponential, no additional dependency."""
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = v / theta
    skew = np.array([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0],
    ], dtype=np.float64)
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * skew
        + (1.0 - np.cos(theta)) * (skew @ skew)
    )


def _rotation_angle(R: np.ndarray) -> float:
    # IEEE Sensors GTSAM Pose3 factors compare rotations on SO(3).
    # The arccos trace formula is numerically safe away from machine zero;
    # for tiny differences any angular error is well below the frozen sigma.
    trace = float(np.trace(R))
    return float(np.arccos(np.clip((trace-1.0)/2.0, -1.0, 1.0)))


def measurement_consistency(
    *,
    gps_local,
    imu_gyro,
    timestamps_seconds,
    lidar_between,
    lidar_valid,
    lidar_quality,
    settings: CrossModalSettings = CrossModalSettings(),
):
    """Compute current-frame-aligned GPS and IMU quality from LiDAR.

    For relative factor between i and i+1, the consistency value is stored
    at target frame i+1 (identical to frozen Stage-7 IMU factor indexing).
    Missing/invalid LiDAR support produces mask=False and q=1 placeholder,
    which MUST NOT be treated as real evidence by the calibration code.
    """
    gps = np.asarray(gps_local, dtype=np.float64)
    gyro = np.asarray(imu_gyro, dtype=np.float64)
    ts = np.asarray(timestamps_seconds, dtype=np.float64).reshape(-1)
    between = np.asarray(lidar_between, dtype=np.float64)
    valid = np.asarray(lidar_valid, dtype=bool).reshape(-1)
    quality = np.asarray(lidar_quality, dtype=np.float64).reshape(-1)
    if gps.ndim != 2 or gps.shape[1] != 3 or len(gps) < 3:
        raise ValueError("gps_local must be N x 3 with N>=3")
    n = len(gps)
    if (gyro.shape != (n, 3) or ts.shape != (n,)
            or between.shape != (n-1, 4, 4)
            or valid.shape != (n-1,) or quality.shape != (n-1,)):
        raise ValueError("GPS, IMU, timestamps and LiDAR factors length/shape mismatch")
    if not all(np.isfinite(v).all() for v in (gps, gyro, ts, between, quality)):
        raise ValueError("Nonfinite input to cross-modal consistency")
    if np.any((quality < 0) | (quality > 1)):
        raise ValueError("LiDAR quality outside [0,1]")
    if settings.gps_window >= n:
        raise ValueError("gps_window must be smaller than total frame count")

    valid_lidar = valid & (quality >= settings.lidar_quality_min)
    gps_q = np.ones(n, dtype=np.float64)
    imu_q = np.ones(n, dtype=np.float64)
    gps_mask = np.zeros(n, dtype=bool)
    imu_mask = np.zeros(n, dtype=bool)
    gps_nis = np.full(n, np.nan, dtype=np.float64)
    imu_nis = np.full(n, np.nan, dtype=np.float64)

    # For 3-D GPS position noise of sigma_g, the difference of two
    # independent position observations has variance 2*sigma_g^2.
    # The LiDAR component is a rough k-step random-walk approximation.
    gps_denom2 = (
        2.0 * settings.gps_sigma_m**2
        + settings.gps_window * settings.lidar_translation_sigma_m**2
    )
    imu_denom2 = (
        settings.imu_rotation_sigma_rad**2
        + settings.lidar_rotation_sigma_rad**2
    )
    for t in range(1, n):
        if valid_lidar[t-1]:
            dt = ts[t] - ts[t-1]
            if np.isfinite(dt) and 0.0 < dt <= 1.0:
                Rimu = _rotvec_to_matrix(gyro[t-1]*dt)
                Rlidar = between[t-1, :3, :3]
                if not np.allclose(
                    Rlidar.T @ Rlidar, np.eye(3), atol=1e-3,
                ) or np.linalg.det(Rlidar) < 0.99:
                    raise ValueError(f"LiDAR rotation not SO(3) at pair {t-1}")
                angle = _rotation_angle(Rimu.T @ Rlidar)
                imu_nis[t] = angle**2 / imu_denom2
                imu_mask[t] = True

        if t < settings.gps_window:
            continue
        start = t - settings.gps_window
        if not np.all(valid_lidar[start:t]):
            continue
        T = np.eye(4)
        for pair in range(start, t):
            T = T @ between[pair]
        delta_gps = float(np.linalg.norm(gps[t] - gps[start]))
        delta_lidar = float(np.linalg.norm(T[:3, 3]))
        gps_nis[t] = (delta_gps - delta_lidar)**2 / gps_denom2
        gps_mask[t] = True

    gps_q[gps_mask] = normalized_innovation_quality(
        gps_nis[gps_mask], settings.gps_chi2_95,
    ) if gps_mask.any() else np.array([], dtype=float)
    imu_q[imu_mask] = normalized_innovation_quality(
        imu_nis[imu_mask], settings.imu_chi2_95,
    ) if imu_mask.any() else np.array([], dtype=float)
    return {
        "gps_quality": gps_q,
        "imu_quality": imu_q,
        "gps_valid": gps_mask,
        "imu_valid": imu_mask,
        "gps_nis": gps_nis,
        "imu_nis": imu_nis,
    }
