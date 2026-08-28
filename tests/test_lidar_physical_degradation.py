import numpy as np
from src.degradation.lidar_physical_degradation import build_schedule,apply_degradation,gt_between,factor_error

def _points(n=1000):
    rng=np.random.default_rng(1); xyz=rng.normal(size=(n,3)); xyz[:,0]+=10
    return np.c_[xyz,rng.random(n)]

def test_schedule_contains_four_types():
    s=build_schedule(3000)
    assert {0,1,2,3,4}.issubset(set(np.unique(s["type_id"])))

def test_sparse_reduces_points():
    p=_points()
    assert len(apply_degradation(p,1,3,10))<len(p)

def test_noise_changes_points():
    p=_points(); q=apply_degradation(p,2,3,10)
    assert q.shape==p.shape and not np.allclose(q[:,:3],p[:,:3])

def test_identity_gt_and_error():
    poses=np.repeat(np.eye(4)[None],3,axis=0)
    b=gt_between(poses)
    assert np.allclose(b,np.repeat(np.eye(4)[None],2,axis=0))
    assert factor_error(np.eye(4),np.eye(4))==(0.0,0.0)
