from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from experiments.isj_supplement.audit_dev_lidar_rotation_reference import (
    rpy_to_matrix,reference_imu_rotation_pairs,relative_rotation_angle_rad,
    relative_yaw_increments,signed_yaw_difference,report,
    resolve_sequence_dir,main,
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



def test_resolve_raw_sequence_strictly_requires_0027(tmp_path):
    wrong=tmp_path/"2011_09_30_drive_0020_sync"
    wrong.mkdir()
    with pytest.raises(ValueError,match="exact drive"):
        resolve_sequence_dir(wrong)
    seq=tmp_path/"2011_10_03_drive_0027_sync"
    seq.mkdir()
    with pytest.raises(FileNotFoundError,match="0027 original OXTS"):
        resolve_sequence_dir(seq)
    (seq/"oxts"/"data").mkdir(parents=True)
    assert resolve_sequence_dir(seq)==seq


def test_auto_mode_falls_back_to_relative_camera_comparison_without_raw_oxts(
    tmp_path,monkeypatch,capsys
):
    from experiments.isj_supplement import audit_dev_lidar_rotation_reference as script
    n=4544
    identity=np.repeat(np.eye(4)[None,:,:],n-1,axis=0)
    fake=SimpleNamespace(
        gps_local=np.zeros((n,3)),
        lidar_between=identity,
        camera_between=identity,
        lidar_valid=np.ones(n-1,bool),
        camera_valid=np.ones(n-1,bool),
        lidar_quality=np.ones(n-1),
        camera_quality=np.ones(n-1),
    )
    monkeypatch.setattr(script.s6,"load_measurements",lambda: fake)
    seq=tmp_path/"2011_10_03_drive_0027_sync"
    main(["--sequence-dir",str(seq),"--reference","auto"])
    printed=capsys.readouterr().out
    assert "OXTS RAW NOT FOUND" in printed
    assert "CAMERA RELATIVE COMPARISON ONLY" in printed
    assert "LIDAR_CAMERA" in printed
    assert "LiDAR versus Camera" in printed
    with pytest.raises(FileNotFoundError,match="0027 original OXTS"):
        main(["--sequence-dir",str(seq),"--reference","oxts"])
