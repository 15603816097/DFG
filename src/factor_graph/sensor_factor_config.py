from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FourSensorFactorConfig:
    """
    Conservative first-pass factor settings.

    These values are NOT the final paper-tuned covariance mapping.
    Their purpose is to validate that the four-sensor factor graph itself
    is correct before any sensor-specific sensitivity sweep.

    Pose3 diagonal sigma order in GTSAM:
        [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    """

    # Global graph settings
    prior_rotation_sigma: float = 1e-3
    prior_translation_sigma: float = 1e-3

    # GPS position pseudo-prior
    gps_fixed_sigma: float = 5.0

    # IMU rotation-only between factor.
    # Translation part is intentionally extremely weak.
    imu_rotation_sigma: float = 0.03
    imu_translation_sigma: float = 1000.0

    # LiDAR relative-pose fixed covariance
    lidar_rotation_sigma: float = 0.03
    lidar_translation_sigma: float = 0.35

    # Camera relative-pose fixed covariance
    camera_rotation_sigma: float = 0.04
    camera_translation_sigma: float = 0.45

    # Reliability-aware provisional ranges.
    # GPS inherits the validated GPS scale.
    gps_sigma_min: float = 3.0
    gps_sigma_max: float = 30.0
    gps_gamma: float = 3.0

    # IMU / LiDAR / Camera are deliberately conservative placeholders.
    # They must be optimized only after the fixed four-sensor graph works.
    imu_rot_sigma_min: float = 0.015
    imu_rot_sigma_max: float = 0.20
    imu_gamma: float = 2.0

    lidar_trans_sigma_min: float = 0.15
    lidar_trans_sigma_max: float = 1.50
    lidar_rot_sigma_min: float = 0.015
    lidar_rot_sigma_max: float = 0.15
    lidar_gamma: float = 2.0

    camera_trans_sigma_min: float = 0.20
    camera_trans_sigma_max: float = 2.00
    camera_rot_sigma_min: float = 0.020
    camera_rot_sigma_max: float = 0.20
    camera_gamma: float = 2.0

    # Quality gating
    lidar_quality_min: float = 0.10
    camera_quality_min: float = 0.10
