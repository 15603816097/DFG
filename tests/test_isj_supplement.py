import numpy as np
import pytest

from experiments.isj_supplement.run_controls import transform_uncertainty


def test_constant_mean_preserves_warmup_and_reliability_independence():
    u = np.array([1.0, 1.0, 0.2, 0.4, 0.8])
    original = u.copy()
    result = transform_uncertainty(u, "constant_mean", 0, "v5", "gps", window=3)
    np.testing.assert_allclose(result, [1.0, 1.0, 0.4666666666666667] * 1
                               + [0.4666666666666667] * 2)
    np.testing.assert_array_equal(u, original)


def test_permuted_is_deterministic_and_keeps_distribution():
    u = np.concatenate([np.ones(2), np.linspace(0.01, 0.98, 31)])
    x1 = transform_uncertainty(u, "permuted", 11, "v6", "imu", window=3)
    x2 = transform_uncertainty(u, "permuted", 11, "v6", "imu", window=3)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(x1[:2], u[:2])
    np.testing.assert_array_equal(np.sort(x1[2:]), np.sort(u[2:]))


def test_bad_uncertainty_rejected():
    with pytest.raises(ValueError):
        transform_uncertainty([1, float("nan"), 0.5], "constant_mean",
                              0, "v5", "camera", window=2)
    with pytest.raises(ValueError):
        transform_uncertainty([1, 2], "permuted",
                              0, "v5", "camera", window=2)
