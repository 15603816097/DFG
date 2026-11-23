from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_dev_odometry_geometry import (
    segment_errors,
)


def fake_data(n=90,step=1.,heading=0.):
    T=np.repeat(np.eye(4)[None,:,:],n-1,axis=0)
    T[:,0,3]=step
    ref=np.column_stack((np.arange(n,dtype=float),np.zeros(n),np.zeros(n)))
    R=np.repeat(np.eye(3)[None,:,:],n,axis=0)
    c,s=np.cos(heading),np.sin(heading)
    R[:,:2,:2]=np.array([[c,-s],[s,c]])
    return T,np.ones(n-1,bool),np.ones(n-1),ref,R


def test_perfect_motion_has_zero_length_vector_heading_error():
    args=fake_data()
    for horizon in (1,5,20):
        stats=segment_errors(*args,horizon,first=8,last=80)
        assert stats["length_mae_m"]==pytest.approx(0.)
        assert stats["vector_mae_m"]==pytest.approx(0.)
        assert stats["heading_abs_median_deg"]==pytest.approx(0.)


def test_wrong_global_heading_does_not_change_norm_error():
    args=fake_data(heading=np.pi/2)
    scores=segment_errors(*args,1,first=5)
    assert scores["length_mae_m"]==pytest.approx(0.)
    assert scores["vector_mae_m"]==pytest.approx(np.sqrt(2))
    assert scores["heading_abs_median_deg"]==pytest.approx(90.)


def test_scale_error_increases_norm_and_vector_error():
    args=fake_data(step=2.)
    scores=segment_errors(*args,5,first=5)
    assert scores["length_mae_m"]==pytest.approx(5.)
    assert scores["vector_mae_m"]==pytest.approx(5.)


def test_missing_lidar_masks_past_dependent_pairs():
    args=list(fake_data())
    args[1][15]=False
    full=segment_errors(*fake_data(),5,first=5)
    smaller=segment_errors(*args,5,first=5)
    assert smaller["pairs"]==full["pairs"]-5


def test_invalid_shape_fails_closed():
    args=list(fake_data())
    args[4]=np.eye(3)
    with pytest.raises(ValueError,match="dimensions"):
        segment_errors(*args,1)
