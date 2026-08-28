import numpy as np


def reliability_to_sigma(
    reliability,
    sigma_min=3.0,
    sigma_max=20.0,
    gamma=3.0,
):
    """
    Continuous reliability-to-covariance mapping.

    Compared with the previous mapping:
        sigma_max: 30 -> 20
        gamma:     2  -> 3

    This preserves useful GPS information for moderate reliability and
    only strongly downweights measurements when reliability becomes low.
    """
    r = np.asarray(
        reliability,
        dtype=np.float64,
    )

    r = np.clip(
        r,
        0.0,
        1.0,
    )

    sigma = (
        sigma_min
        +
        (
            1.0
            -
            r
        )
        ** gamma
        *
        (
            sigma_max
            -
            sigma_min
        )
    )

    return np.clip(
        sigma,
        sigma_min,
        sigma_max,
    )
