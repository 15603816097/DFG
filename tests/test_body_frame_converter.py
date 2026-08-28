from __future__ import annotations

import numpy as np

from src.odometry.body_frame_converter import (
    conjugate_relative_transform,
    coordinate_change_to_between_measurement,
    invert_transform,
    project_to_se3,
)


def make_transform(
    translation,
):
    T = np.eye(
        4,
        dtype=np.float64,
    )
    T[
        :3,
        3
    ] = np.asarray(
        translation,
        dtype=np.float64,
    )
    return T


def test_identity_extrinsic_preserves_coordinate_transform():
    T = make_transform(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    converted = conjugate_relative_transform(
        T,
        np.eye(
            4
        ),
    )

    assert np.allclose(
        converted,
        T,
    )


def test_between_measurement_is_inverse_of_coordinate_change():
    T = make_transform(
        [
            -1.0,
            0.5,
            2.0,
        ]
    )

    between = coordinate_change_to_between_measurement(
        T
    )

    assert np.allclose(
        between,
        invert_transform(
            T
        ),
    )


def test_conjugation_matches_formula():
    sensor_motion = make_transform(
        [
            0.3,
            -0.1,
            0.2,
        ]
    )

    extrinsic = make_transform(
        [
            1.0,
            0.2,
            -0.4,
        ]
    )

    expected = (
        invert_transform(
            extrinsic
        )
        @
        sensor_motion
        @
        extrinsic
    )

    actual = conjugate_relative_transform(
        sensor_motion,
        extrinsic,
    )

    assert np.allclose(
        actual,
        project_to_se3(
            expected
        ),
    )
