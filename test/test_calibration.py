import numpy as np

from src.preprocessing.calibration import (
    KITTICalibration,
)


CALIB_ROOT = (
    "dataset/kitti/"
    "2011_10_03"
)


def test_calibration_load():
    """
    Test loading all KITTI calibration files.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    assert calibration is not None

    print(
        "Calibration root:",
        calibration.calibration_root,
    )


def test_imu_to_velo_transform():
    """
    Test IMU -> LiDAR calibration.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    print(
        "R_imu_to_velo:"
    )

    print(
        calibration.R_imu_to_velo
    )

    print(
        "t_imu_to_velo:"
    )

    print(
        calibration.t_imu_to_velo
    )

    print(
        "T_imu_to_velo:"
    )

    print(
        calibration.T_imu_to_velo
    )

    assert (
        calibration.R_imu_to_velo.shape
        == (3, 3)
    )

    assert (
        calibration.t_imu_to_velo.shape
        == (3,)
    )

    assert (
        calibration.T_imu_to_velo.shape
        == (4, 4)
    )


def test_velo_to_cam_transform():
    """
    Test LiDAR -> Camera calibration.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    print(
        "R_velo_to_cam:"
    )

    print(
        calibration.R_velo_to_cam
    )

    print(
        "t_velo_to_cam:"
    )

    print(
        calibration.t_velo_to_cam
    )

    print(
        "T_velo_to_cam:"
    )

    print(
        calibration.T_velo_to_cam
    )

    assert (
        calibration.R_velo_to_cam.shape
        == (3, 3)
    )

    assert (
        calibration.t_velo_to_cam.shape
        == (3,)
    )

    assert (
        calibration.T_velo_to_cam.shape
        == (4, 4)
    )


def test_composed_imu_to_cam():
    """
    Test composed IMU -> Camera transformation.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    expected = (
        calibration.T_velo_to_cam
        @ calibration.T_imu_to_velo
    )

    assert np.allclose(
        calibration.T_imu_to_cam,
        expected,
    )

    assert (
        calibration.T_imu_to_cam.shape
        == (4, 4)
    )


def test_inverse_transform():
    """
    Test inverse transformation matrices.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    identity_imu = (
        calibration.T_imu_to_velo
        @ calibration.T_velo_to_imu
    )

    identity_velo = (
        calibration.T_velo_to_cam
        @ calibration.T_cam_to_velo
    )

    assert np.allclose(
        identity_imu,
        np.eye(4),
        atol=1e-8,
    )

    assert np.allclose(
        identity_velo,
        np.eye(4),
        atol=1e-8,
    )


def test_camera_projection():
    """
    Test camera projection matrices.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    for camera_id in (
        "image_00",
        "image_01",
        "image_02",
        "image_03",
    ):

        try:

            P = calibration.get_camera_projection(
                camera_id
            )

            print(
                camera_id,
                "projection:"
            )

            print(P)

            assert P.shape == (3, 4)

        except KeyError:

            # Some calibration files may not contain
            # all camera projection matrices.
            print(
                f"{camera_id} projection "
                f"is not available."
            )


def test_camera_intrinsics():
    """
    Test extraction of camera intrinsic matrix.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    K = calibration.get_camera_intrinsics(
        "image_02"
    )

    print(
        "Camera intrinsic matrix:"
    )

    print(K)

    assert K.shape == (3, 3)

    assert np.isfinite(K).all()


def test_transform_points():
    """
    Test transformation of 3D points.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    transformed = (
        calibration.transform_points(
            points,
            calibration.T_imu_to_velo,
        )
    )

    print(
        "Original points:"
    )

    print(points)

    print(
        "Transformed points:"
    )

    print(transformed)

    assert transformed.shape == (
        3,
        3,
    )

    assert np.isfinite(
        transformed
    ).all()


def test_transform_lidar_points():
    """
    Test transformation of KITTI LiDAR-style
    N x 4 points.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    points = np.array(
        [
            [10.0, 0.0, 0.0, 0.5],
            [10.0, 1.0, 0.0, 0.6],
            [10.0, 0.0, 1.0, 0.7],
        ],
        dtype=np.float32,
    )

    transformed = (
        calibration.transform_points(
            points,
            calibration.T_velo_to_cam,
        )
    )

    assert transformed.shape == (
        3,
        3,
    )

    assert np.isfinite(
        transformed
    ).all()


def test_calibration_validation():
    """
    Test calibration matrix quality.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    result = calibration.validate()

    print(
        "Calibration validation:"
    )

    print(result)

    assert "valid" in result

    assert (
        "imu_rotation_error"
        in result
    )

    assert (
        "velo_rotation_error"
        in result
    )

    assert np.isfinite(
        result["imu_rotation_error"]
    )

    assert np.isfinite(
        result["velo_rotation_error"]
    )


def test_lidar_to_camera_projection():
    """
    Test LiDAR point projection into image coordinates.

    We use several forward-facing points. The test focuses
    on numerical validity rather than requiring every point
    to fall inside the camera image.
    """

    calibration = KITTICalibration(
        CALIB_ROOT
    )

    points = np.array(
        [
            [10.0, 0.0, 0.0, 0.5],
            [15.0, 1.0, 0.6, 0.6],
            [20.0, -1.0, 1.0, 0.7],
        ],
        dtype=np.float32,
    )

    result = (
        calibration.project_lidar_to_camera(
            points,
            camera_id="image_02",
        )
    )

    assert (
        "points_camera"
        in result
    )

    assert (
        "pixels"
        in result
    )

    assert (
        "depth"
        in result
    )

    assert (
        "valid_mask"
        in result
    )

    assert (
        result["points_camera"].shape
        == (3, 3)
    )

    assert (
        result["pixels"].shape
        == (3, 2)
    )

    assert (
        result["depth"].shape
        == (3,)
    )

    assert (
        result["valid_mask"].shape
        == (3,)
    )
