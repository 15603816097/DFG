from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_dev_uncertainty_risk_by_split import (
    finite_corr, rank_auc, ranks_with_ties, split_ranges,
    test_risk_metrics as score_risk,
)


def test_v5_and_v6_match_original_chronological_splits_and_exclude_tail():
    n = 4544
    v5 = split_ranges("GRU_V5", n)
    v6 = split_ranges("MAMBA_V6", n)
    assert v5 == {
        "TRAIN": (63, 3180),
        "VAL": (3180, 3862),
        "TEST": (3862, 4542),
    }
    assert v6 == {
        "TRAIN": (63, int(4542*.70)),
        "VAL": (int(4542*.70), int(4542*.85)),
        "TEST": (int(4542*.85), 4542),
    }
    with pytest.raises(ValueError):
        split_ranges("unknown", n)


def test_average_tie_ranking_and_auc():
    assert np.allclose(ranks_with_ties([1,2,2,4]), [1,2.5,2.5,4])
    assert rank_auc([0,1,2,3], [False,False,True,True]) == pytest.approx(1.0)
    assert rank_auc([0,1,2,3], [True,True,False,False]) == pytest.approx(0.0)
    assert rank_auc([1,1,1,1], [False,False,True,True]) == pytest.approx(.5)


def test_simple_risk_rank_correlations():
    u = np.arange(10, dtype=float)
    err = np.arange(10, dtype=float)
    pear, spear, auc = score_risk(u, err)
    assert pear == pytest.approx(1.0)
    assert spear == pytest.approx(1.0)
    assert auc == pytest.approx(1.0)
    assert np.isnan(finite_corr(np.ones(10), err))


def test_noninformative_constant_and_invalid_inputs():
    assert np.isnan(rank_auc([0.,0.], [True,True]))
    assert np.isnan(rank_auc([], []))
    with pytest.raises(ValueError):
        rank_auc([.2,.3], [True])
    with pytest.raises(ValueError):
        ranks_with_ties(np.eye(2))
