from __future__ import annotations

import os
import sys

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


from src.loader.lidar_loader import LidarLoader
from src.odometry.lidar_odometry import (
    LidarOdometry,
)


DATASET = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "lidar_odometry",
)


def extract_points(
    item,
):
    if isinstance(
        item,
        np.ndarray,
    ):
        return item

    if isinstance(
        item,
        dict,
    ):
        for key in (
            "points",
            "point_cloud",
            "lidar",
            "data",
        ):
            if key in item:
                return np.asarray(
                    item[
                        key
                    ]
                )

    raise KeyError(
        "Cannot extract LiDAR points from LidarLoader output"
    )


def main():
    loader = LidarLoader(
        DATASET
    )

    estimator = LidarOdometry(
        max_iterations=20,
        max_correspondence_distance=1.5,
        min_correspondences=80,
        max_points=6000,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    transforms = []
    quality = []

    previous_transform = (
        np.eye(
            4,
            dtype=np.float64,
        )
    )

    print("=" * 88)
    print(
        "KITTI LIDAR RELATIVE ODOMETRY"
    )
    print("=" * 88)

    print(
        "Frames:",
        len(
            loader
        ),
    )

    for i in range(
        len(
            loader
        )
        -
        1
    ):
        source = extract_points(
            loader[
                i
            ]
        )

        target = extract_points(
            loader[
                i + 1
            ]
        )

        # Use previous relative motion only after the first successful frame.
        initial = (
            previous_transform
            if i > 0
            else
            np.eye(
                4,
                dtype=np.float64,
            )
        )

        result = estimator.estimate(
            source,
            target,
            initial_transform=
                initial,
            source_frame_id=
                i,
            target_frame_id=
                i + 1,
        )

        transforms.append(
            result.transform
        )

        quality.append(
            [
                float(
                    result.fitness
                ),
                float(
                    result.rmse
                ),
                float(
                    result.correspondences
                ),
                1.0
                if result.converged
                else 0.0,
            ]
        )

        if result.converged:
            previous_transform = (
                result.transform
            )
        else:
            previous_transform = (
                np.eye(
                    4,
                    dtype=np.float64,
                )
            )

        if i % 100 == 0:
            print(
                f"Frame {i:04d} -> {i+1:04d} | "
                f"fitness={result.fitness:.3f} "
                f"rmse={result.rmse:.3f} "
                f"corr={result.correspondences} "
                f"ok={result.converged}"
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
            "quality.txt",
        ),
        quality,
        fmt="%.8f",
        header=
            "fitness rmse correspondences success",
    )

    # Flatten for easy inspection.
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

    print()
    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
