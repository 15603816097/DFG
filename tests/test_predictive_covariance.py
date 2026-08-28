import numpy as np

from src.factor_graph.predictive_covariance import (
    reliability_to_sigma,
    combine_prediction_with_measurement_quality,
)


def test_sigma_bounds():
    reliability = np.asarray(
        [
            0.0,
            0.5,
            1.0,
        ]
    )

    sigma = reliability_to_sigma(
        reliability,
        3.0,
        30.0,
        3.0,
    )

    assert np.all(
        sigma >= 3.0
    )

    assert np.all(
        sigma <= 30.0
    )

    assert np.isclose(
        sigma[
            0
        ],
        30.0,
    )

    assert np.isclose(
        sigma[
            -1
        ],
        3.0,
    )


def test_sigma_monotonic():
    reliability = np.linspace(
        0.0,
        1.0,
        100,
    )

    sigma = reliability_to_sigma(
        reliability,
        1.0,
        10.0,
        2.0,
    )

    assert np.all(
        np.diff(
            sigma
        )
        <=
        1e-12
    )


def test_quality_combination():
    r = combine_prediction_with_measurement_quality(
        0.8,
        0.5,
    )

    assert np.isclose(
        r,
        0.4,
    )
