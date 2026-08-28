from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VisualOdometryResult:
    transform: np.ndarray
    inliers: int
    matches: int
    reprojection_error: float
    success: bool


def _parse_kitti_calibration(
    calib_path,
):
    """
    Parse KITTI calib_cam_to_cam.txt.

    KITTI calibration files contain both numeric entries:

        P_rect_02: ...
        P_rect_03: ...

    and non-numeric metadata, for example:

        calib_time: 09-Jan-2012 13:57:47

    We therefore only keep lines whose values can be completely
    converted to float arrays.
    """

    calib_path = Path(
        calib_path
    )

    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: "
            f"{calib_path}"
        )

    values = {}

    with calib_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            if ":" not in line:
                continue

            key, raw_value = line.split(
                ":",
                1,
            )

            key = key.strip()

            raw_value = raw_value.strip()

            if not raw_value:
                continue

            tokens = raw_value.split()

            numeric_values = []

            numeric_line = True

            for token in tokens:

                try:
                    numeric_values.append(
                        float(
                            token
                        )
                    )

                except ValueError:

                    numeric_line = False

                    break

            # Ignore metadata such as:
            #
            # calib_time: 09-Jan-2012 13:57:47
            #
            if not numeric_line:
                continue

            values[
                key
            ] = np.asarray(
                numeric_values,
                dtype=np.float64,
            )

    return values


class StereoVisualOdometry:
    """
    KITTI stereo visual odometry.

    Pipeline
    --------

    image_02(t)
        +
    image_03(t)
        ->
    Stereo disparity
        ->
    Metric 3D points

    image_02(t)
        ->
    ORB features
        ->
    temporal matching
        ->
    image_02(t+1)

    3D(t) + 2D(t+1)
        ->
    PnP RANSAC
        ->
    relative camera motion

    Returned transform
    ------------------

    T_(t+1)_t

    meaning:

        p_(t+1)
        =
        T_(t+1)_t
        *
        p_t

    """

    def __init__(
        self,
        calib_cam_to_cam_path,
        num_features=2500,
        min_matches=30,
        min_inliers=20,
        ratio_test=0.75,
        disparity_num=128,
        disparity_block_size=5,
        max_depth=80.0,
    ):

        try:

            import cv2

        except ImportError as exc:

            raise ImportError(
                "StereoVisualOdometry requires OpenCV.\n"
                "Install it with:\n"
                "pip install opencv-python"
            ) from exc

        self.cv2 = cv2

        # ============================================================
        # Load KITTI calibration
        # ============================================================

        calibration = (
            _parse_kitti_calibration(
                calib_cam_to_cam_path
            )
        )

        required_keys = [
            "P_rect_02",
            "P_rect_03",
        ]

        for key in required_keys:

            if key not in calibration:

                available_keys = sorted(
                    calibration.keys()
                )

                raise KeyError(
                    f"Missing calibration key: "
                    f"{key}\n"
                    f"Available numeric keys:\n"
                    f"{available_keys}"
                )

        if calibration[
            "P_rect_02"
        ].size != 12:

            raise ValueError(
                "P_rect_02 must contain "
                "12 numeric values"
            )

        if calibration[
            "P_rect_03"
        ].size != 12:

            raise ValueError(
                "P_rect_03 must contain "
                "12 numeric values"
            )

        P2 = calibration[
            "P_rect_02"
        ].reshape(
            3,
            4,
        )

        P3 = calibration[
            "P_rect_03"
        ].reshape(
            3,
            4,
        )

        # ============================================================
        # Camera intrinsics
        # ============================================================

        self.fx = float(
            P2[
                0,
                0
            ]
        )

        self.fy = float(
            P2[
                1,
                1
            ]
        )

        self.cx = float(
            P2[
                0,
                2
            ]
        )

        self.cy = float(
            P2[
                1,
                2
            ]
        )

        if (
            self.fx <= 0.0
            or
            self.fy <= 0.0
        ):

            raise ValueError(
                "Invalid camera focal length"
            )

        # ============================================================
        # Stereo baseline
        # ============================================================
        #
        # Projection matrix:
        #
        # P =
        # [
        #   fx 0 cx Tx
        #   0 fy cy Ty
        #   0 0 1  0
        # ]
        #
        # Camera center offset:
        #
        # tx = Tx / fx
        #
        # baseline = |tx3 - tx2|
        #
        # ============================================================

        tx2 = float(
            P2[
                0,
                3
            ]
            /
            P2[
                0,
                0
            ]
        )

        tx3 = float(
            P3[
                0,
                3
            ]
            /
            P3[
                0,
                0
            ]
        )

        self.baseline = abs(
            tx3
            -
            tx2
        )

        if self.baseline <= 1e-6:

            raise ValueError(
                f"Invalid stereo baseline: "
                f"{self.baseline}"
            )

        # ============================================================
        # Camera matrix
        # ============================================================

        self.K = np.asarray(
            [
                [
                    self.fx,
                    0.0,
                    self.cx,
                ],
                [
                    0.0,
                    self.fy,
                    self.cy,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        # ============================================================
        # Parameters
        # ============================================================

        self.num_features = int(
            num_features
        )

        self.min_matches = int(
            min_matches
        )

        self.min_inliers = int(
            min_inliers
        )

        self.ratio_test = float(
            ratio_test
        )

        self.max_depth = float(
            max_depth
        )

        # ============================================================
        # ORB
        # ============================================================

        self.orb = (
            self.cv2.ORB_create(
                nfeatures=
                    self.num_features,

                scaleFactor=
                    1.2,

                nlevels=
                    8,

                edgeThreshold=
                    19,

                patchSize=
                    31,

                fastThreshold=
                    15,
            )
        )

        self.matcher = (
            self.cv2.BFMatcher(
                self.cv2.NORM_HAMMING,
                crossCheck=False,
            )
        )

        # ============================================================
        # Stereo SGBM
        # ============================================================

        disparity_num = int(
            disparity_num
        )

        disparity_num = max(
            16,
            (
                disparity_num
                //
                16
            )
            *
            16,
        )

        block_size = int(
            disparity_block_size
        )

        if block_size < 3:
            block_size = 3

        if block_size % 2 == 0:
            block_size += 1

        self.stereo = (
            self.cv2.StereoSGBM_create(

                minDisparity=
                    0,

                numDisparities=
                    disparity_num,

                blockSize=
                    block_size,

                P1=
                    8
                    *
                    block_size
                    *
                    block_size,

                P2=
                    32
                    *
                    block_size
                    *
                    block_size,

                disp12MaxDiff=
                    1,

                uniquenessRatio=
                    10,

                speckleWindowSize=
                    100,

                speckleRange=
                    2,

                preFilterCap=
                    31,

                mode=
                    self.cv2
                    .STEREO_SGBM_MODE_SGBM_3WAY,
            )
        )

    # ================================================================
    # Image utilities
    # ================================================================

    def _gray(
        self,
        image,
    ):

        image = np.asarray(
            image
        )

        if image.ndim == 2:

            return image

        if (
            image.ndim == 3
            and
            image.shape[
                2
            ]
            == 3
        ):

            return (
                self.cv2.cvtColor(
                    image,
                    self.cv2.COLOR_BGR2GRAY,
                )
            )

        if (
            image.ndim == 3
            and
            image.shape[
                2
            ]
            == 4
        ):

            return (
                self.cv2.cvtColor(
                    image,
                    self.cv2.COLOR_BGRA2GRAY,
                )
            )

        raise ValueError(
            f"Unsupported image shape: "
            f"{image.shape}"
        )

    # ================================================================
    # Stereo disparity
    # ================================================================

    def compute_disparity(
        self,
        left,
        right,
    ):

        left_gray = self._gray(
            left
        )

        right_gray = self._gray(
            right
        )

        if (
            left_gray.shape
            !=
            right_gray.shape
        ):

            raise ValueError(
                "Left and right stereo "
                "images must have same size"
            )

        disparity = (
            self.stereo.compute(
                left_gray,
                right_gray,
            )
            .astype(
                np.float32
            )
            /
            16.0
        )

        return disparity

    # ================================================================
    # Stereo 3D point
    # ================================================================

    def _point_from_disparity(
        self,
        u,
        v,
        disparity,
    ):

        x = int(
            round(
                float(
                    u
                )
            )
        )

        y = int(
            round(
                float(
                    v
                )
            )
        )

        height, width = (
            disparity.shape[
                :2
            ]
        )

        if (
            x < 0
            or x >= width
            or y < 0
            or y >= height
        ):

            return None

        d = float(
            disparity[
                y,
                x
            ]
        )

        if not np.isfinite(
            d
        ):

            return None

        # Very small disparity produces
        # unreliable huge depth.
        if d <= 1.0:

            return None

        Z = (
            self.fx
            *
            self.baseline
            /
            d
        )

        if (
            not np.isfinite(
                Z
            )
            or
            Z <= 0.1
            or
            Z > self.max_depth
        ):

            return None

        X = (
            (
                float(
                    u
                )
                -
                self.cx
            )
            *
            Z
            /
            self.fx
        )

        Y = (
            (
                float(
                    v
                )
                -
                self.cy
            )
            *
            Z
            /
            self.fy
        )

        point = np.asarray(
            [
                X,
                Y,
                Z,
            ],
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(
                point
            )
        ):

            return None

        return point

    # ================================================================
    # Visual odometry
    # ================================================================

    def estimate(
        self,
        left_t,
        right_t,
        left_t1,
    ):

        cv2 = self.cv2

        gray_t = self._gray(
            left_t
        )

        gray_t1 = self._gray(
            left_t1
        )

        # ------------------------------------------------------------
        # ORB
        # ------------------------------------------------------------

        keypoints_t, descriptors_t = (
            self.orb.detectAndCompute(
                gray_t,
                None,
            )
        )

        keypoints_t1, descriptors_t1 = (
            self.orb.detectAndCompute(
                gray_t1,
                None,
            )
        )

        if (
            descriptors_t is None
            or
            descriptors_t1 is None
        ):

            return VisualOdometryResult(
                transform=
                    np.eye(
                        4,
                        dtype=np.float64,
                    ),

                inliers=
                    0,

                matches=
                    0,

                reprojection_error=
                    float(
                        "inf"
                    ),

                success=
                    False,
            )

        # ------------------------------------------------------------
        # Temporal feature matching
        # ------------------------------------------------------------

        raw_matches = (
            self.matcher.knnMatch(
                descriptors_t,
                descriptors_t1,
                k=2,
            )
        )

        good_matches = []

        for pair in raw_matches:

            if len(
                pair
            ) < 2:
                continue

            m = pair[
                0
            ]

            n = pair[
                1
            ]

            if (
                m.distance
                <
                self.ratio_test
                *
                n.distance
            ):

                good_matches.append(
                    m
                )

        if len(
            good_matches
        ) < self.min_matches:

            return VisualOdometryResult(
                transform=
                    np.eye(
                        4,
                        dtype=np.float64,
                    ),

                inliers=
                    0,

                matches=
                    len(
                        good_matches
                    ),

                reprojection_error=
                    float(
                        "inf"
                    ),

                success=
                    False,
            )

        # ------------------------------------------------------------
        # Stereo depth at frame t
        # ------------------------------------------------------------

        disparity = (
            self.compute_disparity(
                left_t,
                right_t,
            )
        )

        object_points = []

        image_points = []

        for match in good_matches:

            u, v = keypoints_t[
                match.queryIdx
            ].pt

            point3d = (
                self._point_from_disparity(
                    u,
                    v,
                    disparity,
                )
            )

            if point3d is None:
                continue

            target_uv = keypoints_t1[
                match.trainIdx
            ].pt

            object_points.append(
                point3d
            )

            image_points.append(
                target_uv
            )

        if len(
            object_points
        ) < self.min_matches:

            return VisualOdometryResult(
                transform=
                    np.eye(
                        4,
                        dtype=np.float64,
                    ),

                inliers=
                    0,

                matches=
                    len(
                        object_points
                    ),

                reprojection_error=
                    float(
                        "inf"
                    ),

                success=
                    False,
            )

        object_points = np.asarray(
            object_points,
            dtype=np.float32,
        )

        image_points = np.asarray(
            image_points,
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # PnP RANSAC
        # ------------------------------------------------------------

        success, rvec, tvec, inliers = (
            cv2.solvePnPRansac(

                object_points,

                image_points,

                self.K,

                None,

                flags=
                    cv2.SOLVEPNP_EPNP,

                reprojectionError=
                    2.5,

                confidence=
                    0.999,

                iterationsCount=
                    200,
            )
        )

        if (
            not success
            or
            inliers is None
            or
            len(
                inliers
            )
            <
            self.min_inliers
        ):

            return VisualOdometryResult(
                transform=
                    np.eye(
                        4,
                        dtype=np.float64,
                    ),

                inliers=
                    0
                    if inliers is None
                    else
                    int(
                        len(
                            inliers
                        )
                    ),

                matches=
                    int(
                        len(
                            object_points
                        )
                    ),

                reprojection_error=
                    float(
                        "inf"
                    ),

                success=
                    False,
            )

        # ------------------------------------------------------------
        # Convert to homogeneous transform
        # ------------------------------------------------------------

        R, _ = (
            cv2.Rodrigues(
                rvec
            )
        )

        t = tvec.reshape(
            3
        )

        T = np.eye(
            4,
            dtype=np.float64,
        )

        T[
            :3,
            :3
        ] = R

        T[
            :3,
            3
        ] = t

        # ------------------------------------------------------------
        # Reprojection quality
        # ------------------------------------------------------------

        inlier_index = (
            inliers.reshape(
                -1
            )
        )

        projected, _ = (
            cv2.projectPoints(

                object_points[
                    inlier_index
                ],

                rvec,

                tvec,

                self.K,

                None,
            )
        )

        projected = (
            projected.reshape(
                -1,
                2,
            )
        )

        reprojection_error = float(
            np.mean(
                np.linalg.norm(
                    projected
                    -
                    image_points[
                        inlier_index
                    ],
                    axis=1,
                )
            )
        )

        return VisualOdometryResult(
            transform=
                T,

            inliers=
                int(
                    len(
                        inlier_index
                    )
                ),

            matches=
                int(
                    len(
                        object_points
                    )
                ),

            reprojection_error=
                reprojection_error,

            success=
                True,
        )
