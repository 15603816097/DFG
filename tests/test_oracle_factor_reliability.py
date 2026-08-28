import numpy as np

from src.reliability.oracle_factor_reliability import (
    error_to_reliability,
    relative_pose_error,
)


def test_error_to_reliability_monotonic():
    error = np.asarray(
        [
            0.0,
            1.0,
            2.0,
            3.0,
        ]
    )

    reliability = error_to_reliability(
        error,
        scale=2.0,
    )

    assert np.all(
        np.diff(
            reliability
        )
        <=
        1e-12
    )

    assert reliability[
        0
    ] > reliability[
        -1
    ]


def test_identical_pose_has_zero_error():
    T = np.eye(
        4,
        dtype=np.float64,
    )

    translation_error, rotation_error = (
        relative_pose_error(
            T,
            T,
        )
    )

    assert np.isclose(
        translation_error,
        0.0,
    )

    assert np.isclose(
        rotation_error,
        0.0,
    )


def test_translation_error_detected():
    measured = np.eye(
        4,
        dtype=np.float64,
    )

    reference = np.eye(
        4,
        dtype=np.float64,
    )

    reference[
        0,
        3
    ] = 1.0

    translation_error, rotation_error = (
        relative_pose_error(
            measured,
            reference,
        )
    )

    assert np.isclose(
        translation_error,
        1.0,
    )

    assert np.isclose(
        rotation_error,
        0.0,
    )
