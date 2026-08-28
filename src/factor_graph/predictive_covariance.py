from __future__ import annotations

import numpy as np


def reliability_to_sigma(
    reliability,
    sigma_min,
    sigma_max,
    gamma=2.0,
):
    """
    Map reliability r in [0, 1] to standard deviation.

        sigma(r)
        =
        sigma_min
        +
        (sigma_max - sigma_min) * (1-r)^gamma

    High reliability -> small sigma -> strong factor
    Low reliability  -> large sigma -> weak factor
    """
    r = np.clip(
        np.asarray(
            reliability,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    sigma = (
        float(
            sigma_min
        )
        +
        (
            float(
                sigma_max
            )
            -
            float(
                sigma_min
            )
        )
        *
        (
            1.0
            -
            r
        )
        **
        float(
            gamma
        )
    )

    return np.clip(
        sigma,
        float(
            sigma_min
        ),
        float(
            sigma_max
        ),
    )


def combine_prediction_with_measurement_quality(
    predicted_reliability,
    measurement_quality,
):
    """
    Conservative reliability composition.

    A learned sensor-health prediction cannot make a geometrically bad
    measurement strong, so use:

        effective reliability = prediction * quality

    Both inputs are clipped into [0,1].
    """
    prediction = np.clip(
        np.asarray(
            predicted_reliability,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    quality = np.clip(
        np.asarray(
            measurement_quality,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    return (
        prediction
        *
        quality
    )
