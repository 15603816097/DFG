from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(
        0,
        ROOT,
    )


from src.preprocessing.calibration import KITTICalibration
from src.odometry.body_frame_converter import BodyFrameConverter
from src.odometry.relative_factor_data import (
    make_factor_data,
    normalize_quality,
)


DATE_DIR = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
)

LIDAR_DIR = os.path.join(
    ROOT,
    "results",
    "lidar_odometry",
)

CAMERA_DIR = os.path.join(
    ROOT,
    "results",
    "stereo_visual_odometry",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "body_relative_odometry",
)


def load_matrix(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )
    return np.load(
        path,
        allow_pickle=False,
    )


def load_quality(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )
    return np.loadtxt(
        path,
        dtype=np.float64,
        comments="#",
    )


def build_lidar_quality(table):
    # Columns:
    # fitness rmse correspondences success
    fitness = table[:, 0]
    rmse = table[:, 1]
    correspondences = table[:, 2]
    success = table[:, 3] > 0.5

    valid = (
        success
        &
        np.isfinite(fitness)
        &
        np.isfinite(rmse)
        &
        (fitness > 0.0)
        &
        (correspondences >= 80.0)
    )

    fitness_q = np.clip(
        fitness,
        0.0,
        1.0,
    )

    corr_q = normalize_quality(
        correspondences,
        valid,
    )

    rmse_score = np.zeros_like(
        rmse,
        dtype=np.float64,
    )

    finite = (
        valid
        &
        np.isfinite(rmse)
    )

    rmse_score[finite] = np.exp(
        -rmse[finite]
        /
        0.75
    )

    quality = (
        0.50 * fitness_q
        +
        0.25 * corr_q
        +
        0.25 * rmse_score
    )

    quality[~valid] = 0.0

    return (
        valid,
        np.clip(
            quality,
            0.0,
            1.0,
        ),
    )


def build_camera_quality(table):
    # Columns:
    # inliers matches reprojection_error success
    inliers = table[:, 0]
    matches = table[:, 1]
    reproj = table[:, 2]
    success = table[:, 3] > 0.5

    inlier_ratio = np.divide(
        inliers,
        np.maximum(
            matches,
            1.0,
        ),
    )

    valid = (
        success
        &
        np.isfinite(reproj)
        &
        (inliers >= 20.0)
        &
        (matches >= 30.0)
    )

    inlier_q = np.clip(
        inlier_ratio,
        0.0,
        1.0,
    )

    count_q = normalize_quality(
        inliers,
        valid,
    )

    reproj_score = np.zeros_like(
        reproj,
        dtype=np.float64,
    )

    finite = (
        valid
        &
        np.isfinite(reproj)
    )

    reproj_score[finite] = np.exp(
        -reproj[finite]
        /
        2.0
    )

    quality = (
        0.50 * inlier_q
        +
        0.20 * count_q
        +
        0.30 * reproj_score
    )

    quality[~valid] = 0.0

    return (
        valid,
        np.clip(
            quality,
            0.0,
            1.0,
        ),
    )


def convert_sequence(
    transforms,
    valid,
    quality,
    converter_fn,
):
    coordinate = []
    between = []

    for i in range(
        len(transforms)
    ):
        motion = converter_fn(
            transforms[i],
            valid=bool(
                valid[i]
            ),
            quality=float(
                quality[i]
            ),
        )

        coordinate.append(
            motion.coordinate_transform
        )

        between.append(
            motion.between_measurement
        )

    return make_factor_data(
        np.asarray(
            coordinate,
            dtype=np.float64,
        ),
        np.asarray(
            between,
            dtype=np.float64,
        ),
        valid,
        quality,
    )


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    lidar_transforms = load_matrix(
        os.path.join(
            LIDAR_DIR,
            "relative_transforms.npy",
        )
    )

    camera_transforms = load_matrix(
        os.path.join(
            CAMERA_DIR,
            "relative_transforms.npy",
        )
    )

    lidar_quality_table = load_quality(
        os.path.join(
            LIDAR_DIR,
            "quality.txt",
        )
    )

    camera_quality_table = load_quality(
        os.path.join(
            CAMERA_DIR,
            "quality.txt",
        )
    )

    n = min(
        len(
            lidar_transforms
        ),
        len(
            camera_transforms
        ),
        len(
            lidar_quality_table
        ),
        len(
            camera_quality_table
        ),
    )

    if n <= 0:
        raise RuntimeError(
            "No relative odometry data found"
        )

    lidar_transforms = lidar_transforms[:n]
    camera_transforms = camera_transforms[:n]
    lidar_quality_table = lidar_quality_table[:n]
    camera_quality_table = camera_quality_table[:n]

    calibration = KITTICalibration(
        DATE_DIR
    )

    converter = BodyFrameConverter(
        calibration
    )

    lidar_valid, lidar_quality = (
        build_lidar_quality(
            lidar_quality_table
        )
    )

    camera_valid, camera_quality = (
        build_camera_quality(
            camera_quality_table
        )
    )

    lidar_data = convert_sequence(
        lidar_transforms,
        lidar_valid,
        lidar_quality,
        converter.lidar_to_body,
    )

    camera_data = convert_sequence(
        camera_transforms,
        camera_valid,
        camera_quality,
        converter.camera_to_body,
    )

    lidar_path = os.path.join(
        OUTPUT_DIR,
        "lidar_factor_data.npz",
    )

    camera_path = os.path.join(
        OUTPUT_DIR,
        "camera_factor_data.npz",
    )

    lidar_data.save(
        lidar_path
    )

    camera_data.save(
        camera_path
    )

    # Compatibility files requested for direct inspection.
    np.savez_compressed(
        os.path.join(
            OUTPUT_DIR,
            "lidar_relative_body.npz",
        ),
        coordinate_transforms=
            lidar_data.coordinate_transforms,
        between_measurements=
            lidar_data.between_measurements,
        valid=
            lidar_data.valid,
        quality=
            lidar_data.quality,
    )

    np.savez_compressed(
        os.path.join(
            OUTPUT_DIR,
            "camera_relative_body.npz",
        ),
        coordinate_transforms=
            camera_data.coordinate_transforms,
        between_measurements=
            camera_data.between_measurements,
        valid=
            camera_data.valid,
        quality=
            camera_data.quality,
    )

    print("=" * 96)
    print(
        "KITTI LIDAR / CAMERA -> IMU BODY RELATIVE MOTION"
    )
    print("=" * 96)
    print(
        "Relative pairs:",
        n,
    )
    print(
        "LiDAR valid:",
        int(
            np.sum(
                lidar_data.valid
            )
        ),
        "/",
        n,
    )
    print(
        "Camera valid:",
        int(
            np.sum(
                camera_data.valid
            )
        ),
        "/",
        n,
    )
    print(
        "Both valid:",
        int(
            np.sum(
                lidar_data.valid
                &
                camera_data.valid
            )
        ),
        "/",
        n,
    )
    print()
    print(
        "IMPORTANT CONVENTION"
    )
    print(
        "coordinate_transforms : p_body(t+1) = T @ p_body(t)"
    )
    print(
        "between_measurements  : factor-graph pose increment = inverse(T)"
    )
    print()
    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
