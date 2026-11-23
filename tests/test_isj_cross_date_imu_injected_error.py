from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_cross_date_imu_injected_error import (
    WINDOW, aligned_series, injected_imu_errors,
)


def test_known_gyro_and_acc_injected_errors():
    n = 100
    zero = np.zeros((n,3))
    dg = zero.copy()
    da = zero.copy()
    dg[:,0] = 0.03
    da[:,2] = 0.8
    eg, ea = injected_imu_errors(zero, dg, zero, da)
    assert np.allclose(eg, 0.03)
    assert np.allclose(ea, 0.8)


def test_fail_closed_bad_raw_arrays():
    x = np.zeros((100, 3))
    with pytest.raises(ValueError):
        injected_imu_errors(x, np.zeros((99,3)), x, x)


def test_same_frame_and_factor_aligned_indices():
    n = 100
    err = np.arange(n, dtype=float)
    r = np.linspace(0.2, 0.7, n)
    u = np.linspace(0.8, 0.3, n)
    same_err, same_imp, same_u = aligned_series(err,r,u,0)
    fact_err, fact_imp, fact_u = aligned_series(err,r,u,1)
    assert len(same_err) == len(fact_err) == n-WINDOW+1
    assert same_err[0] == 63
    assert fact_err[0] == 62
    assert same_err[-1] == 99
    assert fact_err[-1] == 98
    assert np.array_equal(same_imp, fact_imp)
    assert np.array_equal(same_u, fact_u)
    with pytest.raises(ValueError):
        aligned_series(err,r,u,2)
