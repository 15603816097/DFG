import numpy as np


def residual_reliability(
    residual,
    scale=8.0,
):
    """
    Convert a position innovation residual into reliability.
    """
    residual = np.asarray(
        residual,
        dtype=np.float64,
    )

    scale = max(
        float(scale),
        1e-6,
    )

    reliability = np.exp(
        -0.5
        *
        (
            residual
            /
            scale
        )
        ** 2
    )

    return np.clip(
        reliability,
        0.0,
        1.0,
    )


def adaptive_fuse_reliability(
    predicted,
    feedback,
    predicted_uncertainty=None,
    base_prediction_weight=0.75,
    min_prediction_weight=0.55,
    max_prediction_weight=0.90,
):
    """
    Adaptive feed-forward + feedback fusion.

    When the predictor is confident, more weight is given to predictive
    reliability.  When predicted reliability is uncertain/moderate,
    online factor-graph innovation has more influence.

    If no explicit uncertainty is available, uncertainty is estimated
    from distance to 0.5:
        confidence = 2 * |r_pred - 0.5|
    """
    predicted = np.clip(
        np.asarray(
            predicted,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    feedback = np.clip(
        np.asarray(
            feedback,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    if predicted_uncertainty is None:
        confidence = (
            2.0
            *
            np.abs(
                predicted
                -
                0.5
            )
        )

        confidence = np.clip(
            confidence,
            0.0,
            1.0,
        )
    else:
        uncertainty = np.clip(
            np.asarray(
                predicted_uncertainty,
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )

        confidence = (
            1.0
            -
            uncertainty
        )

    weight = (
        min_prediction_weight
        +
        confidence
        *
        (
            max_prediction_weight
            -
            min_prediction_weight
        )
    )

    weight = np.clip(
        weight,
        min_prediction_weight,
        max_prediction_weight,
    )

    # Keep global prior around the requested base.
    weight = (
        0.5
        *
        weight
        +
        0.5
        *
        base_prediction_weight
    )

    fused = (
        weight
        *
        predicted
        +
        (
            1.0
            -
            weight
        )
        *
        feedback
    )

    return (
        np.clip(
            fused,
            0.0,
            1.0,
        ),
        weight,
    )
