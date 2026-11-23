from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_cross_date_prediction_quality import (
    WINDOW, beta_invalid, load_frozen, load_raw,
    pearson, spearman, tied_ranks,
)


def test_tied_rank_correlation_and_constant_signal():
    assert np.array_equal(
        tied_ranks(np.array([3, 1, 1, 4.])),
        [3., 1.5, 1.5, 4.]
    )
    x = np.array([1., 2., 3., 4.])
    assert pearson(x, x) == pytest.approx(1.)
    assert spearman(x, x) == pytest.approx(1.)
    assert pearson(x, -x) == pytest.approx(-1.)
    assert np.isnan(spearman(x, np.ones(len(x))))
    with pytest.raises(ValueError):
        pearson(np.arange(3), np.arange(4))


def test_beta_invalid_pair():
    r = np.array([.9, .6, .5])
    u = np.array([.6, .4, 1.0])
    assert beta_invalid(r, u) == pytest.approx(1/3)


def test_raw_schema_and_injected_gps_error(tmp_path):
    n = WINDOW + 7
    clean = np.zeros((n, 3))
    degraded = clean.copy()
    degraded[:, 0] = 3.0
    degraded[:, 1] = 4.0
    np.savez(
        tmp_path / "sensor_data.npz",
        clean_gps_local=clean,
        degraded_gps_local=degraded,
        gps_severity=np.linspace(0, 1, n),
        imu_severity=np.zeros(n),
        camera_severity=np.ones(n),
    )
    data, count = load_raw(tmp_path)
    assert count == n
    assert np.allclose(data["gps_injected_error_m"], 5.0)


def test_frozen_prediction_slice_excludes_first_63(tmp_path):
    n = 279
    folder = tmp_path / "predictions" / "v6"
    folder.mkdir(parents=True)
    r = np.linspace(0.1, 0.9, n)
    u = np.linspace(0.9, 0.1, n)
    np.savetxt(folder / "gps_reliability.txt", r)
    np.savetxt(folder / "gps_uncertainty.txt", u)
    rr, uu = load_frozen(tmp_path, "v6", "gps", n)
    assert len(rr) == 216
    assert rr[0] == pytest.approx(r[63])
    assert uu[-1] == pytest.approx(u[-1])
