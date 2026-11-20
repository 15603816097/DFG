from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_dev_cross_modal_feedback import (
    beta_pair_integrity,
)


def test_valid_beta_head_outputs_pass():
    alpha = np.asarray([1.5, 3.0, 2.0, 7.0])
    beta = np.asarray([2.5, 1.0, 3.0, 4.0])
    r = alpha / (alpha + beta)
    u = 2.0 / (alpha + beta)
    report = beta_pair_integrity(r, u, np.ones(4, dtype=bool))
    assert report["invalid_count"] == 0
    assert report["zero_U_count"] == 0


def test_inconsistent_saved_pairs_are_reported_without_modification():
    r = np.asarray([.8, .9, .5, .7])
    u = np.asarray([.2, .8, .5, .0])
    original_r = r.copy()
    original_u = u.copy()
    report = beta_pair_integrity(r, u, np.ones(4, dtype=bool))
    assert report["invalid_count"] == 2
    assert report["zero_U_count"] == 1
    assert [x["frame"] for x in report["examples"]] == [1,3]
    assert report["max_excess"] == pytest.approx(.6)
    assert np.array_equal(original_r, r)
    assert np.array_equal(original_u, u)


def test_warmup_exclusion_and_rounding_tolerance():
    r = np.asarray([1.0, .8, .9])
    u = np.asarray([1.0, .4 + 5e-6, .2])
    report = beta_pair_integrity(r, u, np.asarray([0, 1, 1], dtype=bool))
    assert report["invalid_count"] == 0
    assert report["total"] == 2


def test_bad_input_and_missing_valid_frames_fails():
    with pytest.raises(ValueError, match="matching 1D"):
        beta_pair_integrity([.5], [.5, .5], [True])
    with pytest.raises(ValueError, match="Invalid r/U"):
        beta_pair_integrity([.5], [np.nan], [True])
    with pytest.raises(ValueError, match="Empty valid"):
        beta_pair_integrity([.5], [.5], [False])
