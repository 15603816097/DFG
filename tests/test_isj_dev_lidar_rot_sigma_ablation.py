from __future__ import annotations

import json
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.isj_supplement import run_dev_lidar_rot_sigma_ablation as mod


def fake_m(n=mod.N_EXPECTED):
    I=np.repeat(np.eye(4)[None],n-1,axis=0)
    return SimpleNamespace(
        gps_local=np.zeros((n,3)),
        lidar_between=I.copy(),camera_between=I.copy(),
        lidar_valid=np.ones(n-1,bool),
        camera_valid=np.ones(n-1,bool),
    )


def test_predeclared_rotation_only_intervention():
    sigmas=np.array([s for _,s in mod.SPECS],float)
    assert np.all(sigmas>0)
    assert np.all(np.diff(sigmas)>0)
    assert sigmas[0]==pytest.approx(mod.s6.LIDAR_R0)
    assert sigmas[1]==pytest.approx(mod.s6.CAM_R0)
    a=mod.spec_for(*mod.SPECS[0])
    b=mod.spec_for(*mod.SPECS[-1])
    differing={k for k in a if a[k]!=b[k]}
    assert differing=={"name","lidar_rot_sigma_rad"}
    assert a["sensor_modes"]=={
        "gps":"fixed","imu":"fixed",
        "lidar":"fixed","camera":"fixed",
    }


def test_audit_has_no_output_or_optimizer(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(mod,"OUT",tmp_path/"absent")
    monkeypatch.setattr(mod.s6,"load_measurements",lambda:fake_m())
    def forbidden(*a,**kw):
        raise AssertionError("audit must never run Stage6 GTSAM")
    monkeypatch.setattr(mod.s6,"run_case",forbidden)
    mod.main(["--stage","audit"])
    assert not mod.OUT.exists()
    output=capsys.readouterr().out
    assert "AUDIT ONLY" in output
    assert "0027" in output


def test_cache_rejects_partial_and_wrong_parameters(tmp_path):
    dest=tmp_path/"CASE"
    sp=mod.spec_for(*mod.SPECS[0])
    assert mod.check_cache(dest,sp,12) is None
    dest.mkdir()
    with pytest.raises(RuntimeError,match="Partial"):
        mod.check_cache(dest,sp,12)
    p=np.repeat(np.eye(4)[None],12,axis=0)
    np.save(dest/"poses.npy",p)
    (dest/"isj_rot_sigma_manifest.json").write_text(json.dumps({
        "spec":sp,"metrics":{"ATE3D":2.3}
    }),encoding="utf-8")
    assert mod.check_cache(dest,sp,12)["metrics"]["ATE3D"]==2.3
    with pytest.raises(ValueError,match="mismatch"):
        mod.check_cache(dest,{**sp,"lidar_rot_sigma_rad":.9},12)
    np.save(dest/"poses.npy",np.eye(4)[None])
    with pytest.raises(ValueError,match="poses"):
        mod.check_cache(dest,sp,12)


def test_execute_patches_only_lidar_rotation_and_restores(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,"OUT",tmp_path/"results")
    m=fake_m(n=100)
    baseline=mod.s6.LIDAR_R0
    evidence=[]
    def stub_stage6_run_case(*,name,measurements,
                             gps_mode,imu_mode,camera_mode,lidar_mode):
        assert measurements is m
        assert {gps_mode,imu_mode,camera_mode,lidar_mode}=={"fixed"}
        evidence.append((name,mod.s6.LIDAR_R0,mod.s6.CAM_R0,
                         mod.s6.LIDAR_T0))
        target=mod.s6.OUT/name
        target.mkdir(parents=True,exist_ok=True)
        np.save(target/"poses.npy",
                np.repeat(np.eye(4)[None],len(m.gps_local),axis=0))
        return {"ATE3D":2.45634,"ATE2D":2.3,"ZRMSE":.5}
    monkeypatch.setattr(mod.s6,"run_case",stub_stage6_run_case)
    name,sigma=mod.SPECS[1]
    result=mod.execute_case(m,name,sigma)
    assert result["ATE3D"]==pytest.approx(2.45634)
    assert evidence==[(name,sigma,mod.s6.CAM_R0,mod.s6.LIDAR_T0)]
    assert mod.s6.LIDAR_R0==baseline
    # Cached replay must never run Stage-6 (nor overwrite the pose).
    monkeypatch.setattr(mod.s6,"run_case",lambda **kwargs:
                        (_ for _ in ()).throw(AssertionError("overwrite")))
    cached=mod.execute_case(m,name,sigma)
    assert cached==result


def test_fixed_0027_frame_count():
    m=fake_m(n=200)
    with pytest.raises(ValueError,match="4544"):
        mod.compare_measurements(m)
