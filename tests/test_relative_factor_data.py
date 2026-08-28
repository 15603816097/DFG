from __future__ import annotations

import numpy as np

from src.odometry.relative_factor_data import (
    make_factor_data,
    motion_disagreement,
    normalize_quality,
)


def test_make_factor_data_shapes():
    n = 5

    coordinate = np.repeat(
        np.eye(
            4,
            dtype=np.float64,
        )[
            None,
            :,
            :
        ],
        n,
        axis=0,
    )

    between = coordinate.copy()

    valid = np.asarray(
        [
            True,
            True,
            False,
            True,
            True,
        ]
    )

    quality = np.linspace(
        0.0,
        1.0,
        n,
    )

    data = make_factor_data(
        coordinate,
        between,
        valid,
        quality,
    )

    assert data.translations.shape == (
        n,
        3,
    )

    assert data.rotations.shape == (
        n,
        3,
        3,
    )

    assert data.valid.shape == (
        n,
    )


def test_normalize_quality_range():
    values = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]
    )

    quality = normalize_quality(
        values
    )

    assert np.all(
        quality >= 0.0
    )

    assert np.all(
        quality <= 1.0
    )


def test_identical_motion_has_zero_disagreement():
    T = np.eye(
        4,
        dtype=np.float64,
    )

    translation_error, rotation_error = (
        motion_disagreement(
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
