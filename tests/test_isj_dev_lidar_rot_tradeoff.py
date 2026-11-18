from __future__ import annotations

import json
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.isj_supplement import audit_dev_lidar_rot_tradeoff as script


def _pose_series(n,dx=0.):
    T=np.repeat(np.eye(4)[None],n,axis=0)
    T[:,0,3]=np.arange(n)*.1+dx
    return T


def _test_measurements(n):
    lidar=np.repeat(np.eye(4)[None],n-1,axis=0)
    camera=lidar.copy()
    ang=np.linspace(0.001,.1,n-1)
    ca,sa=np.cos(ang),np.sin(ang)
    camera[:,0,0]=ca
    camera[:,0,1]=-sa
    camera[:,1,0]=sa
    camera[:,1,1]=ca
    return SimpleNamespace(
        gps_local=np.zeros((n,3)),
        lidar_between=lidar,camera_between=camera,
        lidar_valid=np.ones(n-1,bool),
        camera_valid=np.ones(n-1,bool),
        lidar_quality=np.ones(n-1),
        camera_quality=np.ones(n-1),
    )


def test_angle_quantile_is_sensor_only_and_respects_valid_mask():
    m=_test_measurements(105)
    angle,valid=script.angle_deg_lidar_camera(m)
    assert angle.shape==(104,)
    assert valid.all()
    assert angle[0]<angle[-1]
    assert angle[-1]==pytest.approx(np.rad2deg(.1))
    m.camera_valid[50]=False
    a2,v2=script.angle_deg_lidar_camera(m)
    assert np.array_equal(angle,a2)
    assert not v2[50]


def test_position_metrics_and_rpe_are_distinct():
    n=100
    gt=np.column_stack([np.arange(n)*.1,np.zeros(n),np.zeros(n)])
    pose=_pose_series(n)
    # Constant global offset: high ATE, but exact translational RPE.
    pose[:,0,3]+=2.
    ae3,ae2,zm=script.position_metrics(pose,gt,np.arange(5,90))
    assert ae3==pytest.approx(2.)
    assert ae2==pytest.approx(2.)
    assert zm==pytest.approx(0.)
    assert script.rpe_trans_rmse(
        pose,gt,np.arange(5,90),1
    )==pytest.approx(0.,abs=1e-10)
    assert script.rpe_trans_rmse(
        pose,gt,np.arange(5,80),10
    )==pytest.approx(0.,abs=1e-10)
    with pytest.raises(ValueError,match="start"):
        script.rpe_trans_rmse(pose,gt,np.array([99]),10)


def test_diagnostic_does_not_run_fgo_or_write_files(tmp_path,monkeypatch,capsys):
    n=script.N
    root=tmp_path/"uniform"
    monkeypatch.setattr(script.conv,"OUT",root)
    reference=np.column_stack(
        [np.arange(n)*.1,np.zeros(n),np.zeros(n)]
    )
    path=tmp_path/"reference.txt"
    np.savetxt(path,reference,fmt="%.9f")
    monkeypatch.setattr(script.s6,"GT",path)
    monkeypatch.setattr(script.s6,"load_measurements",
                        lambda:_test_measurements(n))
    monkeypatch.setattr(script.s6,"run_case",lambda *a,**k:
                        (_ for _ in ()).throw(
                            AssertionError("must never optimize")))
    for i,(name,sigma) in enumerate(script.conv.pilot.SPECS):
        folder=root/name
        folder.mkdir(parents=True)
        poses=_pose_series(n,dx=i*.2)
        np.save(folder/"poses.npy",poses)
        (folder/"isj_rot_sigma_convergence_manifest.json").write_text(
            json.dumps({
                "spec":script.conv.desired_spec(name,sigma),
                "continuation":{"stationary_under_rule":True},
                "final_metrics":{"ATE3D":i*.2},
            }),encoding="utf-8"
        )
    original_listing={
        str(f):f.stat().st_mtime_ns for f in root.rglob("*") if f.is_file()
    }
    script.main()
    assert {
        str(f):f.stat().st_mtime_ns for f in root.rglob("*") if f.is_file()
    }==original_listing
    output=capsys.readouterr().out
    assert "HIGH SENSOR DISAGREEMENT" in output
    assert "LOW SENSOR DISAGREEMENT" in output
    assert "Q4" in output
    assert "SENSOR-ONLY CHRONOLOGICAL QUARTER PROFILES" in output
    assert "high_angle_fraction=" in output


def test_missing_or_not_stationary_case_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(script.conv,"OUT",tmp_path/"missing")
    with pytest.raises(FileNotFoundError,match="Missing required"):
        script.get_cases()


def test_sensor_only_profile_contains_no_ground_truth_and_uses_pair_mask():
    m=_test_measurements(120)
    angle,valid=script.angle_deg_lidar_camera(m)
    m.camera_valid[20:25]=False
    angle,valid=script.angle_deg_lidar_camera(m)
    threshold=float(np.percentile(angle[valid],75))
    profile=script.sensor_only_quarter_profile(
        m,angle,valid,threshold,10,100
    )
    assert profile["valid_pairs"]==85
    assert profile["valid_pair_fraction"]==pytest.approx(85/90)
    assert 0<=profile["high_disagreement_fraction"]<=1
    assert profile["angle_p95_deg"]>profile["angle_median_deg"]
    assert profile["lidar_quality_mean"]==pytest.approx(1.)
    assert profile["camera_quality_mean"]==pytest.approx(1.)
    assert profile["lidar_step_distance_mean_m"]==pytest.approx(0.)
    assert profile["camera_step_distance_mean_m"]==pytest.approx(0.)
    with pytest.raises(ValueError,match="bounds"):
        script.sensor_only_quarter_profile(
            m,angle,valid,threshold,-1,100
        )
