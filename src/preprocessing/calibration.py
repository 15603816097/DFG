from pathlib import Path

import numpy as np


class KITTICalibration:
    """
    KITTI calibration loader and transformation manager.

    Expected calibration structure:

        dataset/kitti/2011_10_03/
        ├── calib_cam_to_cam.txt
        ├── calib_imu_to_velo.txt
        └── calib_velo_to_cam.txt

    Main transformations:

        IMU -> LiDAR
        LiDAR -> Camera
        IMU -> Camera
    """

    def __init__(self, calibration_root):
        """
        Parameters
        ----------
        calibration_root : str or Path
            Directory containing KITTI calibration files.
        """

        self.calibration_root = Path(
            calibration_root
        )

        if not self.calibration_root.exists():
            raise FileNotFoundError(
                f"Calibration directory not found: "
                f"{self.calibration_root}"
            )

        if not self.calibration_root.is_dir():
            raise NotADirectoryError(
                f"Calibration root is not a directory: "
                f"{self.calibration_root}"
            )

        self.cam_to_cam_file = (
            self.calibration_root
            / "calib_cam_to_cam.txt"
        )

        self.imu_to_velo_file = (
            self.calibration_root
            / "calib_imu_to_velo.txt"
        )

        self.velo_to_cam_file = (
            self.calibration_root
            / "calib_velo_to_cam.txt"
        )

        self._check_file(
            self.cam_to_cam_file
        )

        self._check_file(
            self.imu_to_velo_file
        )

        self._check_file(
            self.velo_to_cam_file
        )

        self.camera_data = (
            self._load_calibration_file(
                self.cam_to_cam_file
            )
        )

        self.imu_velo_data = (
            self._load_calibration_file(
                self.imu_to_velo_file
            )
        )

        self.velo_cam_data = (
            self._load_calibration_file(
                self.velo_to_cam_file
            )
        )

        self._parse_camera_calibration()

        self._parse_imu_to_velo()

        self._parse_velo_to_cam()

        self.T_imu_to_cam = (
            self.T_velo_to_cam
            @ self.T_imu_to_velo
        )

        self.T_cam_to_imu = np.linalg.inv(
            self.T_imu_to_cam
        )

    # ======================================================
    # Basic file utilities
    # ======================================================

    @staticmethod
    def _check_file(file_path):
        """
        Check whether a calibration file exists.
        """

        if not file_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: "
                f"{file_path}"
            )

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Calibration path is not a file: "
                f"{file_path}"
            )

    @staticmethod
    def _load_calibration_file(file_path):
        """
        Load a KITTI calibration file.

        KITTI calibration format:

            key: value value value ...

        This implementation parses every numeric token
        explicitly, avoiding NumPy fromstring warnings.
        """

        data = {}

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                if ":" not in line:
                    continue

                key, value = line.split(
                    ":",
                    1,
                )

                key = key.strip()
                value = value.strip()

                if not value:
                    continue

                tokens = value.split()

                parsed_values = []

                for token in tokens:
                    try:
                        parsed_values.append(
                            float(token)
                        )
                    except ValueError:
                        continue

                if not parsed_values:
                    continue

                data[key] = np.asarray(
                    parsed_values,
                    dtype=np.float64,
                )

        return data

    # ======================================================
    # Matrix utilities
    # ======================================================

    @staticmethod
    def _make_transform(
        rotation,
        translation,
    ):
        """
        Construct a 4x4 homogeneous transformation.
        """

        rotation = np.asarray(
            rotation,
            dtype=np.float64,
        )

        translation = np.asarray(
            translation,
            dtype=np.float64,
        ).reshape(-1)

        if rotation.shape != (3, 3):
            raise ValueError(
                "Rotation matrix must have shape "
                f"(3, 3), got {rotation.shape}"
            )

        if translation.shape != (3,):
            raise ValueError(
                "Translation must have shape "
                f"(3,), got {translation.shape}"
            )

        transform = np.eye(
            4,
            dtype=np.float64,
        )

        transform[:3, :3] = rotation

        transform[:3, 3] = translation

        return transform

    @staticmethod
    def _reshape_matrix(
        values,
        shape,
        name,
    ):
        """
        Convert a flat calibration array to a matrix.
        """

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        expected_size = (
            shape[0] * shape[1]
        )

        if values.size != expected_size:
            raise ValueError(
                f"{name} expects "
                f"{expected_size} values, "
                f"got {values.size}"
            )

        return values.reshape(shape)

    # ======================================================
    # Camera calibration
    # ======================================================

    def _parse_camera_calibration(self):
        """
        Parse camera calibration parameters.
        """

        self.P_rect_00 = (
            self._get_optional_matrix(
                self.camera_data,
                "P_rect_00",
                (3, 4),
            )
        )

        self.P_rect_01 = (
            self._get_optional_matrix(
                self.camera_data,
                "P_rect_01",
                (3, 4),
            )
        )

        self.P_rect_02 = (
            self._get_optional_matrix(
                self.camera_data,
                "P_rect_02",
                (3, 4),
            )
        )

        self.P_rect_03 = (
            self._get_optional_matrix(
                self.camera_data,
                "P_rect_03",
                (3, 4),
            )
        )

        self.S_rect_00 = (
            self._get_optional_vector(
                self.camera_data,
                "S_rect_00",
                2,
            )
        )

        self.S_rect_01 = (
            self._get_optional_vector(
                self.camera_data,
                "S_rect_01",
                2,
            )
        )

        self.S_rect_02 = (
            self._get_optional_vector(
                self.camera_data,
                "S_rect_02",
                2,
            )
        )

        self.S_rect_03 = (
            self._get_optional_vector(
                self.camera_data,
                "S_rect_03",
                2,
            )
        )

        self.R_rect_00 = (
            self._get_optional_matrix(
                self.camera_data,
                "R_rect_00",
                (3, 3),
            )
        )

        self.R_rect_01 = (
            self._get_optional_matrix(
                self.camera_data,
                "R_rect_01",
                (3, 3),
            )
        )

        self.R_rect_02 = (
            self._get_optional_matrix(
                self.camera_data,
                "R_rect_02",
                (3, 3),
            )
        )

        self.R_rect_03 = (
            self._get_optional_matrix(
                self.camera_data,
                "R_rect_03",
                (3, 3),
            )
        )

    # ======================================================
    # IMU -> LiDAR
    # ======================================================

    def _parse_imu_to_velo(self):
        """
        Parse IMU -> Velodyne calibration.
        """

        if "R" not in self.imu_velo_data:
            raise KeyError(
                "Missing 'R' in "
                "calib_imu_to_velo.txt"
            )

        if "T" not in self.imu_velo_data:
            raise KeyError(
                "Missing 'T' in "
                "calib_imu_to_velo.txt"
            )

        self.R_imu_to_velo = (
            self._reshape_matrix(
                self.imu_velo_data["R"],
                (3, 3),
                "R_imu_to_velo",
            )
        )

        self.t_imu_to_velo = (
            np.asarray(
                self.imu_velo_data["T"],
                dtype=np.float64,
            ).reshape(-1)
        )

        if self.t_imu_to_velo.size != 3:
            raise ValueError(
                "T_imu_to_velo must contain "
                "exactly 3 values"
            )

        self.T_imu_to_velo = (
            self._make_transform(
                self.R_imu_to_velo,
                self.t_imu_to_velo,
            )
        )

        self.T_velo_to_imu = np.linalg.inv(
            self.T_imu_to_velo
        )

    # ======================================================
    # LiDAR -> Camera
    # ======================================================

    def _parse_velo_to_cam(self):
        """
        Parse Velodyne -> Camera calibration.
        """

        if "R" not in self.velo_cam_data:
            raise KeyError(
                "Missing 'R' in "
                "calib_velo_to_cam.txt"
            )

        if "T" not in self.velo_cam_data:
            raise KeyError(
                "Missing 'T' in "
                "calib_velo_to_cam.txt"
            )

        self.R_velo_to_cam = (
            self._reshape_matrix(
                self.velo_cam_data["R"],
                (3, 3),
                "R_velo_to_cam",
            )
        )

        self.t_velo_to_cam = (
            np.asarray(
                self.velo_cam_data["T"],
                dtype=np.float64,
            ).reshape(-1)
        )

        if self.t_velo_to_cam.size != 3:
            raise ValueError(
                "T_velo_to_cam must contain "
                "exactly 3 values"
            )

        self.T_velo_to_cam = (
            self._make_transform(
                self.R_velo_to_cam,
                self.t_velo_to_cam,
            )
        )

        self.T_cam_to_velo = np.linalg.inv(
            self.T_velo_to_cam
        )

    # ======================================================
    # Optional matrix/vector helpers
    # ======================================================

    @staticmethod
    def _get_optional_matrix(
        data,
        key,
        shape,
    ):
        """
        Return a matrix if the key exists.

        If the key is not present, return None.
        """

        if key not in data:
            return None

        return KITTICalibration._reshape_matrix(
            data[key],
            shape,
            key,
        )

    @staticmethod
    def _get_optional_vector(
        data,
        key,
        size,
    ):
        """
        Return a vector if the key exists.

        If the key is not present, return None.
        """

        if key not in data:
            return None

        values = np.asarray(
            data[key],
            dtype=np.float64,
        ).reshape(-1)

        if values.size != size:
            raise ValueError(
                f"{key} expects "
                f"{size} values, "
                f"got {values.size}"
            )

        return values

    # ======================================================
    # Camera intrinsics
    # ======================================================

    def get_camera_projection(
        self,
        camera_id="image_02",
    ):
        """
        Get the projection matrix for a camera.
        """

        projection_map = {
            "image_00": self.P_rect_00,
            "image_01": self.P_rect_01,
            "image_02": self.P_rect_02,
            "image_03": self.P_rect_03,
        }

        if camera_id not in projection_map:
            raise ValueError(
                f"Invalid camera_id: {camera_id}"
            )

        projection = projection_map[
            camera_id
        ]

        if projection is None:
            raise KeyError(
                f"Projection matrix for "
                f"{camera_id} is not available."
            )

        return projection.copy()

    def get_camera_intrinsics(
        self,
        camera_id="image_02",
    ):
        """
        Extract camera intrinsic matrix K.
        """

        P = self.get_camera_projection(
            camera_id
        )

        K = P[:, :3].copy()

        return K

    def get_camera_baseline(
        self,
        camera_id="image_02",
    ):
        """
        Estimate horizontal camera baseline.
        """

        P = self.get_camera_projection(
            camera_id
        )

        fx = P[0, 0]
        tx = P[0, 3]

        if abs(fx) < 1e-12:
            raise ZeroDivisionError(
                "Camera focal length fx is zero."
            )

        baseline = -tx / fx

        return float(baseline)

    # ======================================================
    # Coordinate transformation
    # ======================================================

    @staticmethod
    def transform_points(
        points,
        transform,
    ):
        """
        Transform 3D points using a 4x4 matrix.

        Input:

            (N, 3)

        or:

            (N, 4)

        For KITTI LiDAR:

            x y z intensity
        """

        points = np.asarray(
            points,
            dtype=np.float64,
        )

        transform = np.asarray(
            transform,
            dtype=np.float64,
        )

        if transform.shape != (4, 4):
            raise ValueError(
                "transform must have shape "
                f"(4, 4), got {transform.shape}"
            )

        if points.ndim != 2:
            raise ValueError(
                "points must be a 2D array"
            )

        if points.shape[1] not in (3, 4):
            raise ValueError(
                "points must have shape "
                "(N,3) or (N,4)"
            )

        xyz = points[:, :3]

        ones = np.ones(
            (xyz.shape[0], 1),
            dtype=np.float64,
        )

        homogeneous = np.concatenate(
            [
                xyz,
                ones,
            ],
            axis=1,
        )

        transformed = (
            transform
            @ homogeneous.T
        ).T

        return transformed[:, :3]

    # ======================================================
    # Point projection
    # ======================================================

    def project_lidar_to_camera(
        self,
        points_lidar,
        camera_id="image_02",
        min_depth=1e-6,
    ):
        """
        Project LiDAR points into camera image coordinates.
        """

        points_lidar = np.asarray(
            points_lidar,
            dtype=np.float64,
        )

        if points_lidar.ndim != 2:
            raise ValueError(
                "points_lidar must be a 2D array"
            )

        if points_lidar.shape[1] not in (3, 4):
            raise ValueError(
                "points_lidar must have "
                "3 or 4 columns"
            )

        P = self.get_camera_projection(
            camera_id
        )

        points_camera = self.transform_points(
            points_lidar,
            self.T_velo_to_cam,
        )

        xyz = points_camera

        depth = xyz[:, 2]

        valid_mask = depth > min_depth

        pixels = np.full(
            (xyz.shape[0], 2),
            np.nan,
            dtype=np.float64,
        )

        valid_xyz = xyz[
            valid_mask
        ]

        if valid_xyz.shape[0] > 0:

            homogeneous_points = np.concatenate(
                [
                    valid_xyz,
                    np.ones(
                        (
                            valid_xyz.shape[0],
                            1,
                        ),
                        dtype=np.float64,
                    ),
                ],
                axis=1,
            )

            projected_homogeneous = (
                P
                @ homogeneous_points.T
            ).T

            projected = (
                projected_homogeneous[:, :2]
                /
                projected_homogeneous[:, 2:3]
            )

            pixels[
                valid_mask
            ] = projected

        return {
            "points_camera": points_camera,
            "pixels": pixels,
            "depth": depth,
            "valid_mask": valid_mask,
        }

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def rotation_error(
        rotation,
    ):
        """
        Measure deviation from a valid rotation matrix.
        """

        rotation = np.asarray(
            rotation,
            dtype=np.float64,
        )

        identity = np.eye(
            3,
            dtype=np.float64,
        )

        return float(
            np.linalg.norm(
                rotation.T
                @ rotation
                - identity
            )
        )

    @staticmethod
    def determinant_error(
        rotation,
    ):
        """
        Measure deviation of determinant from 1.
        """

        rotation = np.asarray(
            rotation,
            dtype=np.float64,
        )

        return float(
            abs(
                np.linalg.det(rotation)
                - 1.0
            )
        )

    def validate(self):
        """
        Validate calibration matrices.
        """

        imu_rotation_error = (
            self.rotation_error(
                self.R_imu_to_velo
            )
        )

        velo_rotation_error = (
            self.rotation_error(
                self.R_velo_to_cam
            )
        )

        imu_det_error = (
            self.determinant_error(
                self.R_imu_to_velo
            )
        )

        velo_det_error = (
            self.determinant_error(
                self.R_velo_to_cam
            )
        )

        valid = (
            imu_rotation_error < 1e-5
            and velo_rotation_error < 1e-5
            and imu_det_error < 1e-5
            and velo_det_error < 1e-5
        )

        return {
            "valid": valid,
            "imu_rotation_error":
                imu_rotation_error,
            "velo_rotation_error":
                velo_rotation_error,
            "imu_determinant_error":
                imu_det_error,
            "velo_determinant_error":
                velo_det_error,
        }
