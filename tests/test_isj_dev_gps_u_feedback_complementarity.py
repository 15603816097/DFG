from __future__ import annotations
import numpy as np
import pytest

from experiments.isj_supplement.audit_dev_gps_u_feedback_complementarity import (
    fit_logit_probe,apply_probe,FEATURE_SETS,_sigmoid,
)
from experiments.isj_supplement.audit_dev_uncertainty_risk_by_split import rank_auc


def test_fixed_features_and_stable_sigmoid():
    assert FEATURE_SETS=={
        "R_ONLY":(0,),
        "R_PLUS_D":(0,1),
        "R_PLUS_D_PLUS_U":(0,1,2),
    }
    a=_sigmoid(np.array([-1000.,0.,1000.]))
    assert np.all(np.isfinite(a))
    assert a[0]<1e-100 and a[1]==pytest.approx(.5) and a[2]>1-1e-10


def test_train_only_probe_fits_separable_risk():
    a=np.linspace(-5.,5.,201)
    x=np.column_stack((a,np.cos(a)))
    y=(a>0).astype(int)
    fit=fit_logit_probe(x[:141],y[:141])
    score=apply_probe(fit,x[141:])
    assert score.shape==(60,)
    assert (score>=0).all() and (score<=1).all()
    assert score[-1]>score[0]
    assert fit["coef"][1]>0


def test_probe_is_independent_of_holdout_and_does_not_mutate_inputs():
    rng=np.random.default_rng(108)
    x=rng.normal(size=(180,3))
    y=(x[:,0]+x[:,1]>.4).astype(int)
    train_x=x[:130].copy()
    train_y=y[:130].copy()
    fit=fit_logit_probe(train_x,train_y)
    heldout=x[130:].copy()
    score=apply_probe(fit,heldout)
    # No held-out labels passed to fit or apply; caller can change TEST
    # labels without affecting any score (diagnostic train/val split).
    holdout_labels=y[130:].copy()
    holdout_labels[:]=1-holdout_labels
    assert np.array_equal(score,apply_probe(fit,heldout))
    assert np.array_equal(x[:130],train_x)
    assert np.array_equal(x[130:],heldout)
    assert np.array_equal(y[:130],train_y)


def test_matched_auc_and_degenerate_inputs():
    scores=np.array([0.1,0.2,0.9,1.])
    yy=np.array([0,0,1,1])
    assert rank_auc(scores,yy)==pytest.approx(1.)
    assert np.isnan(rank_auc(scores,np.ones(4,bool)))
    with pytest.raises(ValueError,match="TRAIN"):
        fit_logit_probe(np.ones((22,2)),np.ones(22))
    with pytest.raises(ValueError,match="held-out"):
        apply_probe(fit_logit_probe(
            np.arange(40)[:,None].astype(float),
            (np.arange(40)>20).astype(int)
        ),np.ones((4,2)))
