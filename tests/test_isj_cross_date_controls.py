from __future__ import annotations

import numpy as np
import pytest

from experiments.cross_sequence.run_stage7_factor_graph import confidence
from experiments.isj_supplement.run_cross_date_controls import (
    DEFAULT_SEEDS,
    WINDOW,
    control_specs,
    controlled_prediction,
    parse_seeds,
)
from experiments.isj_supplement.run_controls import transform_uncertainty


def test_predeclared_controls_are_eight_distinct_cases():
    specs = control_specs(("gru", "mamba"), "all", DEFAULT_SEEDS)
    assert len(specs) == 8
    assert len({s["name"] for s in specs}) == 8
    assert {s["source"] for s in specs} == {"v5", "v6"}
    assert sum(s["mode"] == "constant_mean" for s in specs) == 2
    assert sum(s["mode"] == "permuted" for s in specs) == 6
    assert all(s["type"] == "uncertainty" for s in specs)


def test_parse_seeds_and_restrict_models():
    assert parse_seeds("11, 23,47") == (11, 23, 47)
    for item in ("", "11,11", "-1", "a,1"):
        with pytest.raises(ValueError):
            parse_seeds(item)
    with pytest.raises(ValueError):
        control_specs(("unknown",), "all", DEFAULT_SEEDS)
    assert len(control_specs(("mamba",), "constant_mean", DEFAULT_SEEDS)) == 1


def test_temporal_controls_keep_warmup_and_uncertainty_distribution():
    u = np.r_[np.ones(WINDOW - 1), np.linspace(0.2, 0.9, 30)]
    for mode in ("constant_mean", "permuted"):
        transformed = transform_uncertainty(u, mode, 11, "v6", "gps", WINDOW)
        assert np.array_equal(transformed[:WINDOW - 1], u[:WINDOW - 1])
        if mode == "constant_mean":
            assert np.allclose(transformed[WINDOW - 1:], np.mean(u[WINDOW - 1:]))
        else:
            assert np.allclose(
                np.sort(transformed[WINDOW - 1:]), np.sort(u[WINDOW - 1:])
            )
            assert not np.array_equal(transformed[WINDOW - 1:], u[WINDOW - 1:])


def test_controls_never_modify_frozen_reliability():
    n = WINDOW + 16
    r = np.linspace(0.4, 0.9, n)
    u = np.r_[np.ones(WINDOW - 1), np.linspace(0.2, 0.8, n - WINDOW + 1)]

    def original_pred(seq, source, sensor, num):
        assert source == "v5" and sensor == "gps" and num == n
        return r, u

    spec = control_specs(("gru",), "permuted", (23,))[0]
    out_r, out_u = controlled_prediction(original_pred, spec)(
        "2011_09_30_drive_0016_sync", "v5", "gps", n
    )
    assert np.array_equal(out_r, r)
    assert np.array_equal(out_u[:WINDOW - 1], u[:WINDOW - 1])
    assert np.allclose(np.sort(out_u[WINDOW - 1:]), np.sort(u[WINDOW - 1:]))


def test_dynamic_warmup_uses_fixed_confidence_not_reliability_only():
    # Actual solver semantics: dynamic-U uses confidence=1-U, so U=1
    # disables r-based adjustment. Reliability-only forces confidence=1.
    assert confidence(1.0) == 0.0
    assert confidence(0.0) == 1.0
