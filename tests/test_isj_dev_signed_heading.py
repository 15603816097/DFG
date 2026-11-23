from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_dev_signed_heading import (
    historical_yaw,orientation_stream,make_heading_pairs,
    angles_from_vectors,describe_xy_pair,best_oracle_yaw,
)


def sample(n=95,heading=np.pi/4):
    T=np.repeat(np.eye(4)[None,:,:],n-1,axis=0)
    T[:,0,3]=1.0
    mask=np.ones(n-1,bool)
    quality=np.ones(n-1,float)
    gps=np.column_stack([
        np.arange(n)*np.cos(heading),
        np.arange(n)*np.sin(heading),
        np.zeros(n),
    ])
    return gps,T,mask,quality


def test_past_only_yaw_bootstrap_matches_clean_synthetic_trajectory():
    gps,T,mask,quality=sample()
    yaw=historical_yaw(gps,T,mask,quality)
    assert yaw==pytest.approx(np.pi/4)
    R,good=orientation_stream(T,mask,quality,yaw)
    frames,a,b=make_heading_pairs(T,good,R,gps,5)
    assert frames[0]==63
    assert np.allclose(a,b,atol=1e-10)
    stats=describe_xy_pair(a,b)
    assert stats["abs_angle_median_deg"]==pytest.approx(0,abs=1e-8)
    assert stats["vector_mae_m"]==pytest.approx(0,abs=1e-8)


def test_fixed_yaw_offset_is_detected_with_signed_angle_and_oracle():
    gps,T,mask,quality=sample(heading=0)
    # Deliberately use an incorrect initial world alignment +0.5 rad.
    R,good=orientation_stream(T,mask,quality,0.5)
    _,a,b=make_heading_pairs(T,good,R,gps,1)
    d=angles_from_vectors(a,b)
    assert np.median(d)==pytest.approx(-.5)
    oracle=best_oracle_yaw(a,b)
    assert oracle==pytest.approx(-.5)
    stats=describe_xy_pair(a,b,oracle_correction_rad=oracle)
    assert stats["vector_mae_m"]>0.4
    assert stats["ORACLE_vector_mae_m"]==pytest.approx(0,abs=1e-12)


def test_missing_pairs_do_not_leak_forward_or_get_scored():
    gps,T,mask,quality=sample()
    mask[70]=False
    yaw=historical_yaw(gps,T,mask,quality)
    R,good=orientation_stream(T,mask,quality,yaw)
    frames,_,_=make_heading_pairs(T,good,R,gps,5)
    assert not np.any((frames<=70)&(frames+5>70))
    assert 75 in frames
    assert not good[70]


def test_model_bootstrap_and_eval_shapes_validate():
    gps,T,mask,quality=sample()
    with pytest.raises(ValueError,match="history"):
        historical_yaw(gps,T[:40],mask[:40],quality[:40])
    with pytest.raises(ValueError,match="valid"):
        orientation_stream(T,mask[:40],quality,0.)
    with pytest.raises(ValueError,match="diagnostic"):
        make_heading_pairs(T,mask,np.eye(3),gps,1)
    with pytest.raises(ValueError,match="pair"):
        angles_from_vectors(np.zeros((4,2)),np.zeros((3,2)))
