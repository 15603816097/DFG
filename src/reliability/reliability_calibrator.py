import numpy as np


class ReliabilityCalibrator:
    """
    Conservative-but-not-overly-pessimistic reliability calibration.

    Purpose
    -------
    Raw Mamba reliability can preserve temporal trend but may be
    mis-calibrated in absolute value.  This module remaps raw
    reliability into a better calibrated confidence score without
    retraining the temporal model.

    Mapping
    -------
    1. optional affine correction around the raw mean
    2. temperature-style logistic sharpening
    3. clipping

    The default mapping intentionally raises mid/high confidence while
    keeping truly low predictions low.
    """

    def __init__(
        self,
        center=0.55,
        slope=1.35,
        temperature=0.85,
        minimum=0.02,
        maximum=0.995,
    ):
        self.center = float(center)
        self.slope = float(slope)
        self.temperature = float(temperature)
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def transform(self, reliability):
        r = np.asarray(
            reliability,
            dtype=np.float64,
        )

        r = np.clip(
            r,
            1e-6,
            1.0 - 1e-6,
        )

        # Affine expansion around a moderate confidence center.
        r = (
            self.center
            +
            self.slope
            *
            (
                r
                -
                self.center
            )
        )

        r = np.clip(
            r,
            1e-6,
            1.0 - 1e-6,
        )

        # Logistic temperature calibration.
        logit = np.log(
            r
            /
            (
                1.0
                -
                r
            )
        )

        calibrated = (
            1.0
            /
            (
                1.0
                +
                np.exp(
                    -logit
                    /
                    max(
                        self.temperature,
                        1e-6,
                    )
                )
            )
        )

        return np.clip(
            calibrated,
            self.minimum,
            self.maximum,
        )
