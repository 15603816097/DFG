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


from src.odometry.stereo_visual_odometry import (
    StereoVisualOdometry,
)


DATE_DIR = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
)

SEQUENCE = os.path.join(
    DATE_DIR,
    "2011_10_03_drive_0027_sync",
)

LEFT_DIR = os.path.join(
    SEQUENCE,
    "image_02",
    "data",
)

RIGHT_DIR = os.path.join(
    SEQUENCE,
    "image_03",
    "data",
)

CALIB_PATH = os.path.join(
    DATE_DIR,
    "calib_cam_to_cam.txt",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "stereo_visual_odometry",
)


def main():
    try:
        import cv2
    except Exception as exc:
        raise ImportError(
            "OpenCV is required. Install opencv-python if cv2 is missing."
        ) from exc

    left_files = sorted(
        Path(
            LEFT_DIR
        ).glob(
            "*.png"
        )
    )

    right_files = sorted(
        Path(
            RIGHT_DIR
        ).glob(
            "*.png"
        )
    )

    n = min(
        len(
            left_files
        ),
        len(
            right_files
        ),
    )

    if n < 2:
        raise RuntimeError(
            "Stereo image sequence not found"
        )

    estimator = (
        StereoVisualOdometry(
            CALIB_PATH,
            num_features=2500,
            min_matches=30,
            min_inliers=20,
            ratio_test=0.75,
            disparity_num=128,
            disparity_block_size=5,
            max_depth=80.0,
        )
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    transforms = []
    quality = []

    print("=" * 88)
    print(
        "KITTI STEREO VISUAL ODOMETRY"
    )
    print("=" * 88)

    print(
        "Frames:",
        n,
    )

    print(
        "Stereo baseline:",
        estimator.baseline,
    )

    for i in range(
        n - 1
    ):
        left_t = cv2.imread(
            str(
                left_files[
                    i
                ]
            ),
            cv2.IMREAD_COLOR,
        )

        right_t = cv2.imread(
            str(
                right_files[
                    i
                ]
            ),
            cv2.IMREAD_COLOR,
        )

        left_t1 = cv2.imread(
            str(
                left_files[
                    i + 1
                ]
            ),
            cv2.IMREAD_COLOR,
        )

        if (
            left_t is None
            or right_t is None
            or left_t1 is None
        ):
            result_transform = np.eye(
                4,
                dtype=np.float64,
            )

            transforms.append(
                result_transform
            )

            quality.append(
                [
                    0.0,
                    0.0,
                    float(
                        "inf"
                    ),
                    0.0,
                ]
            )

            continue

        result = estimator.estimate(
            left_t,
            right_t,
            left_t1,
        )

        transforms.append(
            result.transform
        )

        quality.append(
            [
                float(
                    result.inliers
                ),
                float(
                    result.matches
                ),
                float(
                    result.reprojection_error
                ),
                1.0
                if result.success
                else 0.0,
            ]
        )

        if i % 100 == 0:
            print(
                f"Frame {i:04d} -> {i+1:04d} | "
                f"inliers={result.inliers} "
                f"matches={result.matches} "
                f"reproj={result.reprojection_error:.3f} "
                f"ok={result.success}"
            )

    transforms = np.asarray(
        transforms,
        dtype=np.float64,
    )

    quality = np.asarray(
        quality,
        dtype=np.float64,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "relative_transforms.npy",
        ),
        transforms,
    )

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "relative_transforms.txt",
        ),
        transforms.reshape(
            len(
                transforms
            ),
            16,
        ),
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "quality.txt",
        ),
        quality,
        fmt="%.8f",
        header=
            "inliers matches reprojection_error success",
    )

    print()
    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
