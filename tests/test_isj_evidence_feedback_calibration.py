from __future__ import annotations

import numpy as np
import pytest

from src.reliability.evidence_feedback_calibration import (
    CalibrationSettings, beta_soft_evidence_update,
    normalized_innovation_quality,
)


def test_quality_cutoff_and_decrease_for_large_innovations():
    c = 2.795**2
    q = normalized_innovation_quality(np.array([0., c, 4*c, 16*c]), c)
    assert np.allclose(q, [1., 1., .5, .25])
    with pytest.raises(ValueError):
        normalized_innovation_quality([-1.], c)
    with pytest.raises(ValueError):
        normalized_innovation_quality([0.], 0.0)


def test_soft_pseudocount_is_beta_posterior_mean():
    r = np.array([.8, .3])
    u = np.array([.2, .2])
    q = np.array([0., 1.])
    out, w = beta_soft_evidence_update(
        r, u, q, settings=CalibrationSettings(2, 0)
    )
    # 2/U=10 effective network evidence counts; add k=2 pseudo counts.
    assert np.allclose(w, 1./6.)
    assert np.allclose(out, [(10*.8+2*0)/12, (10*.3+2*1)/12])
    assert out[0] < r[0]
    assert out[1] > r[1]


def test_larger_valid_U_gives_more_feedback_influence():
    r = np.array([.5, .5])
    u = np.array([.1, .8])
    q = np.array([0., 0.])
    out, weight = beta_soft_evidence_update(
        r, u, q, settings=CalibrationSettings(2, 0)
    )
    assert weight[1] > weight[0] > 0
    assert out[1] < out[0] < .5


def test_mask_and_warmup_do_not_modify_original():
    n = 68
    r = np.full(n, .6)
    u = np.full(n, .5)
    # Before the window fills, r=.5,U=1 in GRU, r=1,U=1 in Mamba;
    # both are artificial placeholders, NOT calibrated beta evidence.
    r[:63] = 1.
    u[:63] = 1.
    q = np.zeros(n)
    valid = np.ones(n, dtype=bool)
    valid[65] = False
    out, weights = beta_soft_evidence_update(
        r, u, q, valid_feedback=valid
    )
    assert np.array_equal(out[:63], r[:63])
    assert np.array_equal(weights[:63], np.zeros(63))
    assert out[65] == r[65]
    assert weights[65] == 0
    assert np.all(out[np.array([63, 64, 66, 67])] < .6)


def test_beta_validity_and_input_checks_fail_closed():
    s = CalibrationSettings(2, 0)
    with pytest.raises(ValueError, match="Beta evidence invalid"):
        beta_soft_evidence_update([.9], [.8], [.2], settings=s)
    with pytest.raises(ValueError, match="Nonfinite"):
        beta_soft_evidence_update([.5], [.5], [float("nan")], settings=s)
    with pytest.raises(ValueError, match="boolean"):
        beta_soft_evidence_update([.5], [.5], [.5],
                                  valid_feedback=np.array([1]), settings=s)
    with pytest.raises(ValueError, match="equal"):
        beta_soft_evidence_update([.5], [.5,.5], [.5], settings=s)
    with pytest.raises(ValueError, match="pseudo_count"):
        CalibrationSettings(pseudo_count=-1)
    with pytest.raises(ValueError, match="expected_warmup"):
        CalibrationSettings(expected_warmup=-1)
