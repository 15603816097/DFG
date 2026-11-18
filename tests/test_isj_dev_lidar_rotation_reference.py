from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_dev_lidar_rotation_reference import (
    rpy_to_matrix,reference_imu_rotation_pairs,relative_rotation_angle_rad,
    relative_yaw_increments,signed_yaw_difference,report,
)


def batch_z(angles):
    return np.stack([rpy_to_matrix(0.,0.,a) for a in angles])


def test_kitti_body_relative_yaw_from_oxts(tmp_path):
    oxts=tmp_path/"oxts"/"data"
    oxts.mkdir(parents=True)
    n=80
    for i in range(n):
        # Fake OXTS [lat,lon,alt,roll,pitch,yaw]; angle increases +.01.
        np.savetxt(oxts/f"{i:010d}.txt",
                   np.array([0.,0.,0.,0.,0.,.01*i]))
    pairs=reference_imu_rotation_pairs(tmp_path,n)
    assert pairs.shape==(n-1,3,3)
    assert np.allclose(relative_yaw_increments(pairs),.01,atol=1e-12)
    with pytest.raises(ValueError,match="Expected"):
        reference_imu_rotation_pairs(tmp_path,n+1)


def test_so3_and_signed_angle_errors_distinguish_bias_and_inverse():
    a=batch_z([.02]*30)
    b=batch_z([.01]*30)
    residual=relative_rotation_angle_rad(a,b)
    assert np.degrees(np.mean(residual))==pytest.approx(
        np.degrees(.01),rel=1e-7
    )
    signed=signed_yaw_difference(a,b)
    assert np.allclose(signed,.01,atol=1e-12)
    inverse=relative_rotation_angle_rad(np.swapaxes(a,1,2),b)
    assert np.allclose(inverse,.03,atol=1e-7)


def test_zero_error_and_bad_dimensions():
    identity=np.repeat(np.eye(3)[None],30,axis=0)
    assert np.max(relative_rotation_angle_rad(identity,identity))<1e-7
    assert np.allclose(signed_yaw_difference(identity,identity),0)
    with pytest.raises(ValueError,match="matching"):
        relative_rotation_angle_rad(identity,identity[:20])
    with pytest.raises(ValueError,match="rotation"):
        relative_yaw_increments(np.eye(3))


def test_report_matched_pairs_and_partial_validity():
    n=300
    l=batch_z(np.full(n-1,.011))
    gt=batch_z(np.full(n-1,.010))
    mask=np.ones(n-1,bool)
    mask[100:200]=False
    error,inverse,yaw=report("LiDAR",l,gt,mask,n)
    assert len(error)>100
    assert len(inverse)==len(error)==len(yaw)
    # report() returns yaw residuals in DEGREES, not radians.
    assert np.mean(yaw)==pytest.approx(np.degrees(.001),abs=1e-12)
    with pytest.raises(ValueError,match="too few|Too few"):
        report("LiDAR",l,gt,np.zeros(n-1,bool),n)
