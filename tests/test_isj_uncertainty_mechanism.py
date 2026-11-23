import numpy as np
import pytest

from experiments.isj_supplement.diagnose_uncertainty_mechanism import (
    compare_error_curves,
    high_low_effect,
    safe_corr,
)


def test_framewise_error_gain_positive_means_dynamic_better():
    gt = np.zeros((4, 3))
    full = np.array([[1., 0, 0], [2., 0, 0],
                     [3., 0, 0], [4., 0, 0]])
    reliability = full + np.array([[1., 0, 0], [0., 0, 0], [-1., 0, 0], [-2., 0, 0]])
    curves = compare_error_curves(gt, full, reliability)
    np.testing.assert_allclose(
        curves["dynamic_gain_per_frame_m"], [1., 0., -1., -2.]
    )


def test_quartile_effect_sign_is_dynamic_minus_reliability():
    u = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
    dynamic = np.ones(8)
    rel = np.ones(8) * 2
    effect = high_low_effect(u, dynamic, rel)
    assert effect["quartiles_distinct"]
    assert effect["low_count"] == 2
    assert effect["high_count"] == 2
    assert effect["low_dynamic_minus_reliability_rmse_m"] == pytest.approx(-1.)
    assert effect["high_dynamic_minus_reliability_rmse_m"] == pytest.approx(-1.)


def test_constant_uncertainty_does_not_make_fake_quartiles():
    effect = high_low_effect(
        np.ones(10) * .5, np.ones(10), np.ones(10) * 2
    )
    assert effect["quartiles_distinct"] is False
    assert effect["low_count"] == 0
    assert np.isnan(effect["low_dynamic_minus_reliability_rmse_m"])


def test_safe_corr_constant_signal_is_undefined():
    assert np.isnan(safe_corr(np.ones(5), np.arange(5)))
    assert safe_corr(np.arange(5), np.arange(5)) == pytest.approx(1.0)


def test_error_comparison_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        compare_error_curves(np.zeros((3, 3)), np.zeros((4, 3)), np.zeros((4, 3)))
