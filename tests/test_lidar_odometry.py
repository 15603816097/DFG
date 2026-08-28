import numpy as np

from src.odometry.lidar_odometry import (
    LidarOdometry,
)


def test_lidar_icp_recovers_translation():
    rng = np.random.default_rng(
        0
    )

    source = rng.normal(
        size=(
            1000,
            3,
        )
    ).astype(
        np.float32
    )

    source[
        :,
        0
    ] += 10.0

    translation = np.asarray(
        [
            0.30,
            -0.10,
            0.05,
        ]
    )

    target = (
        source
        +
        translation
    )

    estimator = LidarOdometry(
        max_iterations=30,
        max_correspondence_distance=1.0,
        min_correspondences=50,
        max_points=1000,
        min_range=0.0,
        max_range=100.0,
        z_min=-100.0,
        z_max=100.0,
    )

    result = estimator.estimate(
        source,
        target,
    )

    assert result.correspondences > 500

    assert np.allclose(
        result.transform[
            :3,
            3
        ],
        translation,
        atol=0.08,
    )


def test_lidar_bad_input():
    estimator = LidarOdometry()

    try:
        estimator.estimate(
            np.zeros(
                (
                    10,
                    2,
                )
            ),
            np.zeros(
                (
                    10,
                    2,
                )
            ),
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
