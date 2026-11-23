from __future__ import annotations

import numpy as np
import pytest

from src.reliability.causal_gps_position_innovation import (
    CausalInnovationSettings,causal_position_innovation,
)


def build(n=90,heading=np.pi/2):
    local=np.arange(n,dtype=float)
    gps=np.column_stack([
        10+np.cos(heading)*local,
        20+np.sin(heading)*local,
        np.zeros(n),
    ])
    lidar=np.repeat(np.eye(4)[None],n-1,axis=0)
    lidar[:,0,3]=1
    camera=lidar.copy()
    return dict(
        gps_local=gps,
        lidar_between=lidar,lidar_valid=np.ones(n-1,bool),
        lidar_quality=np.ones(n-1,float),
        camera_between=camera,camera_valid=np.ones(n-1,bool),
        camera_quality=np.ones(n-1,float),
    )


def test_perfect_past_only_headed_odometry_predicts_future_gps():
    d=build()
    s=causal_position_innovation(**d)
    assert not s["valid"][:63].any()
    assert s["valid"][63:].all()
    assert np.allclose(s["prior_position"][63:],d["gps_local"][63:],atol=1e-5)
    assert np.allclose(s["innovation_norm_m"][63:],0,atol=1e-5)
    assert np.all(s["geometry_consistent"][63:])
    assert np.all(s["odometry_source"][63:]==1)
    assert s["bootstrapped_heading_rad"]==pytest.approx(np.pi/2)


def test_preupdate_prediction_ignores_current_and_future_gps():
    d=build()
    before=causal_position_innovation(**d)
    d["gps_local"][70,1]+=50
    after=causal_position_innovation(**d)
    # 70 prior must be identical. 70 innovation uses 70 GPS measurement.
    assert np.array_equal(
        before["prior_position"][:71],after["prior_position"][:71],
        equal_nan=True,
    )
    assert after["innovation_norm_m"][70]>49
    assert np.array_equal(
        before["innovation"][:70],after["innovation"][:70],
        equal_nan=True,
    )
    # Post-update correction is allowed to affect 71+ priors, but not 70.
    assert np.linalg.norm(
        before["prior_position"][71]-after["prior_position"][71]
    )>0


def test_prior_falls_back_to_camera_and_rejects_inconsistent_geometry():
    d=build()
    d["lidar_valid"][69]=False
    d["lidar_between"][75,0,3]=4
    x=causal_position_innovation(**d)
    assert x["odometry_source"][70]==2
    assert x["valid"][70]
    assert not x["geometry_consistent"][70]  # no two-sensor confirmation
    assert x["odometry_source"][76]==1
    assert not x["geometry_consistent"][76]
    assert x["pair_translation_difference_m"][76]==pytest.approx(3)


def test_no_motion_support_masks_current_not_fake_zero_nis():
    d=build()
    d["lidar_valid"][69]=False
    d["camera_valid"][69]=False
    x=causal_position_innovation(**d)
    assert not x["valid"][70]
    assert np.isnan(x["standardized_score"][70])
    assert x["odometry_source"][70]==0


def test_fail_closed_on_shapes_and_invalid_config():
    d=build()
    d["camera_between"]=np.eye(4)
    with pytest.raises(ValueError,match="Invalid camera"):
        causal_position_innovation(**d)
    with pytest.raises(ValueError):
        CausalInnovationSettings(warmup=2)
    with pytest.raises(ValueError):
        CausalInnovationSettings(gps_sigma_m=-1)
