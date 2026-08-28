from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _as_transform(T, name="transform"):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values")
    return T


def invert_transform(T):
    T = _as_transform(T)
    R = T[:3, :3]
    t = T[:3, 3]

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = R.T
    result[:3, 3] = -(R.T @ t)
    return result


def project_to_se3(T):
    T = _as_transform(T).copy()
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = R
    result[:3, 3] = T[:3, 3]
    return result


def rotation_angle(R):
    R = np.asarray(R, dtype=np.float64)
    value = (np.trace(R) - 1.0) * 0.5
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def transform_angle(T):
    return rotation_angle(_as_transform(T)[:3, :3])


def transform_translation_norm(T):
    return float(np.linalg.norm(_as_transform(T)[:3, 3]))


def conjugate_relative_transform(
    relative_sensor_transform,
    T_sensor_from_body,
):
    """
    Convert a coordinate-change transform from a sensor frame to body/IMU.

    Input convention
    ----------------
    relative_sensor_transform = T_sensor(t+1)_sensor(t)

        p_sensor(t+1)
        =
        relative_sensor_transform
        @
        p_sensor(t)

    Extrinsic convention
    --------------------
    T_sensor_from_body maps body/IMU coordinates into sensor coordinates:

        p_sensor = T_sensor_from_body @ p_body

    Output
    ------
    T_body(t+1)_body(t)

        p_body(t+1)
        =
        T_body(t+1)_body(t)
        @
        p_body(t)

    Formula
    -------
        T_body = inv(E) @ T_sensor @ E
    """
    T_sensor = project_to_se3(relative_sensor_transform)
    E = project_to_se3(T_sensor_from_body)

    return project_to_se3(
        invert_transform(E)
        @ T_sensor
        @ E
    )


def coordinate_change_to_between_measurement(
    coordinate_change,
):
    """
    Convert coordinate-change odometry into a pose-increment measurement.

    The LiDAR ICP and stereo VO in this project return:

        T_(t+1)_t

    which maps coordinates from frame t into frame t+1.

    A factor-graph BetweenFactor Pose3 measurement represents the vehicle
    pose increment from state t to state t+1, so it is the inverse:

        Z_t_t+1 = inv(T_(t+1)_t)
    """
    return project_to_se3(
        invert_transform(
            coordinate_change
        )
    )


def build_camera02_rect_from_imu(calibration):
    """
    Build T_camera02_rect_from_imu for KITTI raw calibration.

    Existing KITTICalibration provides:
        T_imu_to_velo
        T_velo_to_cam        (unrectified camera-0 reference)
        R_rect_00
        P_rect_02

    Stereo VO reconstructs points in rectified image_02 coordinates.
    Therefore the camera extrinsic used for conjugation must include:
      1. IMU -> Velodyne
      2. Velodyne -> camera-0
      3. camera-0 -> rectified camera-0
      4. rectified camera-0 -> rectified camera-2 translation encoded by P2
    """
    if calibration.R_rect_00 is None:
        raise KeyError("R_rect_00 is required for image_02 conversion")
    if calibration.P_rect_02 is None:
        raise KeyError("P_rect_02 is required for image_02 conversion")

    T_rect0_from_cam0 = np.eye(4, dtype=np.float64)
    T_rect0_from_cam0[:3, :3] = calibration.R_rect_00

    P2 = np.asarray(calibration.P_rect_02, dtype=np.float64)
    K2 = P2[:, :3]
    p4 = P2[:, 3]

    try:
        t_cam2_from_rect0 = np.linalg.solve(K2, p4)
    except np.linalg.LinAlgError as exc:
        raise ValueError("P_rect_02 intrinsic block is singular") from exc

    T_cam2rect_from_rect0 = np.eye(4, dtype=np.float64)
    T_cam2rect_from_rect0[:3, 3] = t_cam2_from_rect0

    return project_to_se3(
        T_cam2rect_from_rect0
        @ T_rect0_from_cam0
        @ calibration.T_velo_to_cam
        @ calibration.T_imu_to_velo
    )


@dataclass
class BodyRelativeMotion:
    coordinate_transform: np.ndarray
    between_measurement: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    valid: bool
    quality: float


class BodyFrameConverter:
    def __init__(self, calibration):
        self.calibration = calibration

        # Velodyne coordinates from IMU/body coordinates.
        self.T_lidar_from_body = project_to_se3(
            calibration.T_imu_to_velo
        )

        # Rectified image_02 coordinates from IMU/body coordinates.
        self.T_camera_from_body = build_camera02_rect_from_imu(
            calibration
        )

    @staticmethod
    def _pack(
        coordinate_transform,
        valid,
        quality,
    ):
        coordinate_transform = project_to_se3(
            coordinate_transform
        )

        between = coordinate_change_to_between_measurement(
            coordinate_transform
        )

        return BodyRelativeMotion(
            coordinate_transform=coordinate_transform,
            between_measurement=between,
            translation=between[:3, 3].copy(),
            rotation=between[:3, :3].copy(),
            valid=bool(valid),
            quality=float(quality),
        )

    def lidar_to_body(
        self,
        relative_lidar_transform,
        valid=True,
        quality=1.0,
    ):
        T_body = conjugate_relative_transform(
            relative_lidar_transform,
            self.T_lidar_from_body,
        )

        return self._pack(
            T_body,
            valid,
            quality,
        )

    def camera_to_body(
        self,
        relative_camera_transform,
        valid=True,
        quality=1.0,
    ):
        T_body = conjugate_relative_transform(
            relative_camera_transform,
            self.T_camera_from_body,
        )

        return self._pack(
            T_body,
            valid,
            quality,
        )
