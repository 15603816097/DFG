from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.run_cross_date_warmup_matched import (
    WINDOW, specs_for_models, warmup_matched_prediction,
)
from experiments.cross_sequence import run_stage7_factor_graph as s7


def test_two_unique_isolated_cases():
    cases = specs_for_models(("gru", "mamba"))
    assert len(cases) == 2
    assert len({c["name"] for c in cases}) == 2
    assert {c["source"] for c in cases} == {"v5", "v6"}
    assert all(c["type"] == "dynamic" for c in cases)
    with pytest.raises(ValueError):
        specs_for_models(("bogus",))


@pytest.mark.parametrize("source", ("v5", "v6"))
def test_matched_warmup_reliability_unchanged_and_phase_switch(source):
    n = 279
    r = np.linspace(0.1, 0.9, n)
    u = np.r_[np.ones(WINDOW - 1), np.linspace(0.2, 0.8, n - WINDOW + 1)]
    r_orig = r.copy()
    u_orig = u.copy()

    def pred(seq, src, sensor, count):
        assert src == source and sensor == "gps" and count == n
        return r, u

    wrapped = warmup_matched_prediction(pred, source)
    new_r, new_u = wrapped("2011_09_30_drive_0016_sync", source, "gps", n)
    assert np.array_equal(new_r, r_orig)
    assert np.array_equal(r, r_orig)
    assert np.array_equal(u, u_orig)
    assert np.array_equal(new_u[:WINDOW - 1], u[:WINDOW - 1])
    assert np.array_equal(new_u[WINDOW - 1:], np.zeros(n-WINDOW+1))

    # The solver maps U=1 to confidence=0 (fixed sigma) and U=0 to
    # confidence=1 (the existing Reliability Only predictive sigma).
    assert all(s7.confidence(v) == 0.0 for v in new_u[:WINDOW - 1])
    assert all(s7.confidence(v) == 1.0 for v in new_u[WINDOW - 1:])
    assert np.array_equal(new_u[:WINDOW-1], u[:WINDOW-1])


def test_fail_closed_if_frozen_warmup_not_one():
    n = 279
    r = np.full(n, 0.6)
    u = np.ones(n)
    u[0] = 0.999

    def pred(*args):
        return r, u

    with pytest.raises(ValueError, match="warmup U is not 1.0"):
        warmup_matched_prediction(pred, "v5")("seq", "v5", "imu", n)


def test_fail_closed_wrong_model_or_sensor():
    n = 279

    def pred(*args):
        return np.full(n, 0.5), np.ones(n)

    wrapped = warmup_matched_prediction(pred, "v5")
    with pytest.raises(ValueError):
        wrapped("seq", "v6", "imu", n)
    with pytest.raises(ValueError):
        wrapped("seq", "v5", "lidar", n)
