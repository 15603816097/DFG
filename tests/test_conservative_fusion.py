import numpy as np

from src.reliability.conservative_fusion import (
    align_pair_quality_to_frames,
    fuse_sensor_reliability,
)


def test_pair_quality_target_alignment():
    q = np.asarray(
        [
            0.2,
            0.4,
            0.6,
        ]
    )

    aligned = align_pair_quality_to_frames(
        q,
        4,
    )

    assert np.allclose(
        aligned,
        [
            1.0,
            0.2,
            0.4,
            0.6,
        ],
    )


def test_camera_never_exceeds_online_quality():
    predicted = np.asarray(
        [
            0.99,
            0.90,
            0.20,
            0.80,
        ]
    )

    quality = np.asarray(
        [
            0.30,
            0.50,
            0.80,
            0.20,
        ]
    )

    result = fuse_sensor_reliability(
        "camera",
        predicted,
        quality,
    )

    assert np.all(
        result
        <=
        quality
        +
        1e-12
    )


def test_lidar_prediction_only_small_upward_correction():
    predicted = np.asarray(
        [
            1.0,
            0.0,
            1.0,
        ]
    )

    quality = np.asarray(
        [
            0.20,
            0.80,
            0.50,
        ]
    )

    result = fuse_sensor_reliability(
        "lidar",
        predicted,
        quality,
    )

    assert np.all(
        result
        <=
        quality
        +
        0.05
        +
        1e-12
    )


def test_imu_has_safety_floor():
    result = fuse_sensor_reliability(
        "imu",
        np.zeros(
            100
        ),
    )

    assert np.all(
        result
        >=
        0.30
    )
