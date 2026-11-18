from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.run_cross_date_beta_swap import (
    WINDOW,
    beta_invalid_fraction,
    beta_valid_swap_uncertainty,
    control_predictions,
    specifications,
)


def fixture_predictions():
    # Synthetic Beta-evidential outputs with alpha,beta >= 1.
    n = 279
    r = np.r_[np.full(WINDOW - 1, 1.0),
              np.linspace(0.25, 0.75, n - (WINDOW - 1))]
    r[WINDOW - 1:] = 0.45 + 0.2 * np.sin(np.linspace(0, 9, n - WINDOW + 1))
    # Vary uncertainty while always respecting alpha>=1,beta>=1.
    caps = 2 * np.minimum(r[WINDOW - 1:], 1 - r[WINDOW - 1:])
    u = np.r_[np.ones(WINDOW - 1),
              caps * (0.35 + 0.35 * np.cos(np.linspace(0, 15, n - WINDOW + 1)) ** 2)]
    return r, u


def test_constraints_for_original_and_invalid_pair():
    r, u = fixture_predictions()
    assert beta_invalid_fraction(r[WINDOW - 1:], u[WINDOW - 1:]) == 0
    assert beta_invalid_fraction(np.array([0.9]), np.array([0.6])) == 1.0
    assert beta_invalid_fraction(np.array([0.9]), np.array([0.2])) == 0


@pytest.mark.parametrize("seed", (11, 23, 47))
def test_beta_swap_preserves_r_and_u_multiset_and_warmup(seed):
    r, u = fixture_predictions()
    original_r = r.copy()
    original_u = u.copy()
    result, stats = beta_valid_swap_uncertainty(
        r, u, seed=seed, source="v6", sensor="gps"
    )
    assert np.array_equal(r, original_r)
    assert np.array_equal(u, original_u)
    assert np.array_equal(result[:WINDOW - 1], u[:WINDOW - 1])
    assert np.array_equal(
        np.sort(result[WINDOW - 1:]),
        np.sort(u[WINDOW - 1:])
    )
    assert beta_invalid_fraction(r[WINDOW - 1:], result[WINDOW - 1:]) == 0
    assert stats["changed_fraction"] >= 0.05
    assert stats["controlled_beta_invalid_fraction"] == 0.0
    reproduced, reproduced_stats = beta_valid_swap_uncertainty(
        r, u, seed=seed, source="v6", sensor="gps"
    )
    assert np.array_equal(result, reproduced)
    assert stats == reproduced_stats


def test_control_predictions_source_checked_and_six_unique_specs():
    r, u = fixture_predictions()
    specs = specifications(("gru", "mamba"), (11, 23, 47))
    assert len(specs) == 6
    assert len({spec["name"] for spec in specs}) == 6

    def original_pred(seq, source, sensor, n):
        assert n == 279
        return r, u

    wrapped = control_predictions(original_pred, specs[0])
    rr, uu = wrapped("2011_09_30_drive_0016_sync", "v5", "gps", 279)
    assert np.array_equal(rr, r)
    assert beta_invalid_fraction(rr[WINDOW - 1:], uu[WINDOW - 1:]) == 0
    with pytest.raises(ValueError):
        wrapped("2011_09_30_drive_0016_sync", "v6", "gps", 279)


def test_original_invalid_fails_closed():
    r, u = fixture_predictions()
    u[WINDOW - 1] = 0.999
    with pytest.raises(ValueError, match="original frozen predictions violate"):
        beta_valid_swap_uncertainty(
            r, u, seed=11, source="v5", sensor="imu"
        )
