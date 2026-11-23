from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_dev_uncertainty_risk_by_split import (
    finite_corr, rank_auc, ranks_with_ties, split_ranges,
    test_risk_metrics as score_risk,
    common_val_test_ranges, conditional_added_value,
    fit_val_risk_calibrator, paired_auc_block_ci,
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



def test_common_val_and_test_indices_are_same_for_both_models():
    val,test=common_val_test_ranges(4544)
    assert val==(3180,3860)
    assert test==(3862,4542)
    assert test[0]-val[1]==2


def test_conditionally_informative_u_improves_r_plus_u_on_future_test():
    pytest.importorskip("sklearn")
    n=4544
    r=np.full(n,.8)
    u=(np.arange(n)%100)/100.
    # Synthetic target is not made from GT and only used in this
    # numerical test. It lies in [0,1].
    err=np.where(u>=.8,.3,.01)
    target=r-err
    out=conditional_added_value(
        target,r,u,*common_val_test_ranges(n)
    )
    assert out["val_n"]==680
    assert out["test_n"]==680
    assert 0.15<out["test_positive_fraction"]<.25
    assert out["delta_auc_rplusu_minus_r"]>.45
    assert out["delta_brier_rminus_rplusu"]>.01
    assert out["delta_logloss_rminus_rplusu"]>.03
    assert out["delta_auc_block_ci95"][0]>0


def test_conditional_calibrator_refuses_invalid_training_labels():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError,match="Invalid VAL"):
        fit_val_risk_calibrator(
            np.ones((40,1)),np.zeros(40,bool),np.ones((40,1))
        )
    with pytest.raises(ValueError,match="Invalid ordered"):
        conditional_added_value(
            np.full(100,.7),np.full(100,.8),np.full(100,.2),
            (70,90),(80,100)
        )


def test_identical_score_block_bootstrap_has_zero_auc_delta():
    pytest.importorskip("sklearn")
    n=200
    y=(np.arange(n)%7)==0
    prob=(np.arange(n)%11)/11.
    lo,hi,used=paired_auc_block_ci(
        y,prob,prob,block=16,reps=60,seed=1
    )
    assert used>40
    assert lo==pytest.approx(0.)
    assert hi==pytest.approx(0.)
