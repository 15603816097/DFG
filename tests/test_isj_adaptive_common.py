import math

import numpy as np
import pytest

from experiments.isj_supplement.adaptive_fg_common import (
    CHI_3D_95,
    MAX_SIGMA_MULTIPLIER,
    sigma_multiplier,
)


def test_residual_adaptive_sigma_identity_below_cutoff():
    assert sigma_multiplier([0.0, 0.0, 0.0], CHI_3D_95) == 1.0
    assert sigma_multiplier([CHI_3D_95, 0.0, 0.0], CHI_3D_95) == 1.0


def test_residual_adaptive_sigma_inflates_smoothly_and_caps():
    multiplier = sigma_multiplier([4.0 * CHI_3D_95, 0.0, 0.0], CHI_3D_95)
    assert multiplier == pytest.approx(2.0)
    very_large = sigma_multiplier([1000.0, 0.0, 0.0], CHI_3D_95)
    assert very_large == MAX_SIGMA_MULTIPLIER


def test_residual_adaptive_sigma_rejects_invalid_input():
    with pytest.raises(ValueError):
        sigma_multiplier([float("nan"), 0.0, 0.0], CHI_3D_95)
    with pytest.raises(ValueError):
        sigma_multiplier([1.0, 0.0, 0.0], 0.0)
