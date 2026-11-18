from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from src.reliability.causal_gps_position_innovation import (
    CausalInnovationSettings, causal_position_innovation,
)
from experiments.isj_supplement.audit_dev_e1_motion_policy import (
    POLICIES,
)


def simple(n=180, lidar_yaw_drift=0.025):
    gps=np.column_stack((
        np.arange(n,dtype=float),
        np.zeros(n),
        np.zeros(n),
    ))
    lidar=np.repeat(np.eye(4)[None,:,:],n-1,axis=0)
    camera=lidar.copy()
    lidar[:,0,3]=1.0
    camera[:,0,3]=1.0
    c,s=np.cos(lidar_yaw_drift),np.sin(lidar_yaw_drift)
    lidar[:,:2,:2]=np.array([[c,-s],[s,c]])
    return dict(
        gps_local=gps,
        lidar_between=lidar,lidar_valid=np.ones(n-1,bool),
        lidar_quality=np.ones(n-1,float),
        camera_between=camera,camera_valid=np.ones(n-1,bool),
        camera_quality=np.ones(n-1,float),
    )


def test_default_lidar_first_is_backwards_compatible():
    d=simple()
    baseline=causal_position_innovation(**d)
    explicit=causal_position_innovation(
        **d,settings=CausalInnovationSettings(motion_policy="lidar_first")
    )
    assert baseline["motion_policy"]=="lidar_first"
    for key in ("prior_position","innovation",
                "standardized_score","odometry_source"):
        assert np.array_equal(baseline[key],explicit[key],equal_nan=True)


def test_camera_rot_and_hybrid_fix_synthetic_lidar_angular_drift():
    d=simple()
    outputs={
        policy:causal_position_innovation(
            **d,settings=CausalInnovationSettings(motion_policy=policy)
        )
        for policy in POLICIES
    }
    last=slice(150,180)
    lidar=float(np.nanmean(outputs["lidar_first"]["innovation_norm_m"][last]))
    camera=float(np.nanmean(outputs["camera_first"]["innovation_norm_m"][last]))
    hybrid=float(np.nanmean(
        outputs["camera_rotation_lidar_translation"]["innovation_norm_m"][last]
    ))
    assert camera<lidar*.3
    assert hybrid<lidar*.3
    assert np.all(outputs["camera_first"]["odometry_source"][63:]==2)
    assert np.all(
        outputs["camera_rotation_lidar_translation"]["odometry_source"][63:]==3
    )


def test_fallback_when_camera_unavailable_uses_lidar():
    d=simple(100)
    d["camera_valid"][75]=False
    outputs={
        policy:causal_position_innovation(
            **d,settings=CausalInnovationSettings(motion_policy=policy)
        )
        for policy in ("camera_first","camera_rotation_lidar_translation")
    }
    assert outputs["camera_first"]["odometry_source"][76]==1
    assert outputs["camera_rotation_lidar_translation"]["odometry_source"][76]==1


def test_preupdate_non_leakage_for_camera_and_hybrid():
    for policy in ("camera_first","camera_rotation_lidar_translation"):
        d=simple(100)
        cfg=CausalInnovationSettings(motion_policy=policy)
        old=causal_position_innovation(**d,settings=cfg)
        d["gps_local"][71,1]+=30.0
        new=causal_position_innovation(**d,settings=cfg)
        assert np.array_equal(
            old["prior_position"][:72],new["prior_position"][:72],
            equal_nan=True,
        )
        assert new["innovation_norm_m"][71]>old["innovation_norm_m"][71]+20


def test_invalid_policy_rejected():
    with pytest.raises(ValueError,match="motion_policy"):
        replace(CausalInnovationSettings(),motion_policy="guess")
