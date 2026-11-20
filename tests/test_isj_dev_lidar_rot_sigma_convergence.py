from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.isj_supplement import run_dev_lidar_rot_sigma_convergence as mod


def fake_poses(n):
    poses=np.repeat(np.eye(4)[None],n,axis=0)
    poses[:,0,3]=np.linspace(0,3,n)
    return poses


def test_four_frozen_specs_and_equal_rules():
    specs=[mod.desired_spec(n,s) for n,s in mod.pilot.SPECS]
    assert len(specs)==4
    for s in specs:
        assert s["max_extra_passes"]==4
        assert s["objective_rtol"]==pytest.approx(1e-7)
        assert s["pose_shift_rmse_tol_m"]==pytest.approx(.001)
    for a,b in zip(specs,specs[1:]):
        differing={key for key in a if a[key]!=b[key]}
        assert differing=={"source_case","source_sigma_rad"}


def test_cache_is_fail_closed_on_partial_and_wrong_spec(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,"N",12)
    folder=tmp_path/"OUTPUT"
    spec=mod.desired_spec(*mod.pilot.SPECS[0])
    assert mod.get_cached_result(folder,spec) is None
    folder.mkdir()
    with pytest.raises(RuntimeError,match="Partial"):
        mod.get_cached_result(folder,spec)
    np.save(folder/"poses.npy",fake_poses(12))
    (folder/"isj_rot_sigma_convergence_manifest.json").write_text(
        json.dumps({
            "spec":spec,
            "final_metrics":{"ATE3D":2.3},
        }),
        encoding="utf-8",
    )
    assert mod.get_cached_result(folder,spec)["final_metrics"]["ATE3D"]==2.3
    with pytest.raises(ValueError,match="Conflicting"):
        mod.get_cached_result(folder,{**spec,"source_sigma_rad":.9})
    np.save(folder/"poses.npy",fake_poses(3))
    with pytest.raises(ValueError,match="poses"):
        mod.get_cached_result(folder,spec)


def test_continuation_uses_graph_cost_pose_shift_and_not_eval(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,"N",8)
    # Patch only transform serializer; this is a numerical unit test
    # without requiring the user's GTSAM dependency in the test runner.
    monkeypatch.setattr(mod.s6,"_pose3_to_matrix",lambda M:M)
    monkeypatch.setattr(mod.s6,"_pose3_from_matrix",lambda gtsam,M:M)

    class Values:
        def __init__(self): self.poses={}
        def insert(self,i,matrix):self.poses[i]=matrix
        def atPose3(self,i):return self.poses[i]

    class Params:
        def setMaxIterations(self,n):self.n=n
        def setRelativeErrorTol(self,t):self.t=t

    class Graph:
        def error(self,values):
            return float(sum(
                values.atPose3(i)[0,3]**2 for i in range(8)
            ))

    class ExactSolver:
        def __init__(self,graph,initial,params):
            assert params.n==100
            assert params.t==1e-7
            self.values=initial
        def optimize(self):
            # Already stationary: same physical poses, zero shift and
            # zero relative objective decrease.
            return self.values

    fake=SimpleNamespace(Values=Values,LevenbergMarquardtParams=Params)
    poses=fake_poses(8)
    obj=Graph()
    initial=obj.error(mod.values_from_poses(fake,poses))
    state,hist,stationary,warning=mod.continuation(
        fake,obj,poses,solver_class=ExactSolver,max_extra_passes=4
    )
    assert len(hist)==2
    assert hist[0]["objective"]==pytest.approx(initial)
    assert hist[1]["objective"]==pytest.approx(initial)
    assert hist[1]["position_shift_rmse_m"]==pytest.approx(0.)
    assert stationary
    assert warning is None
    assert np.allclose(mod.poses_from_values(state,8),poses)


def test_nonstationary_continuation_reports_not_stable(monkeypatch):
    monkeypatch.setattr(mod,"N",8)
    monkeypatch.setattr(mod.s6,"_pose3_to_matrix",lambda m:m)
    monkeypatch.setattr(mod.s6,"_pose3_from_matrix",lambda gtsam,m:m)

    class Values:
        def __init__(self): self.data={}
        def insert(self,i,x):self.data[i]=x
        def atPose3(self,i):return self.data[i]

    class Params:
        def setMaxIterations(self,n):pass
        def setRelativeErrorTol(self,x):pass

    class Graph:
        def error(self,vals):
            return sum(vals.atPose3(i)[0,3]**2 for i in range(8))

    class Solver:
        def __init__(self,graph,values,params):self.values=values
        def optimize(self):
            out=Values()
            for i in range(8):
                x=self.values.atPose3(i).copy()
                x[0,3]*=.5
                out.insert(i,x)
            return out

    g=SimpleNamespace(Values=Values,LevenbergMarquardtParams=Params)
    _,history,stationary,warning=mod.continuation(
        g,Graph(),fake_poses(8),
        solver_class=Solver,max_extra_passes=2,
    )
    assert not stationary and warning is None
    assert len(history)==3
    assert history[1]["objective"]<history[0]["objective"]
    assert history[2]["position_shift_rmse_m"]>0


def test_audit_uses_existing_0027_files_and_does_not_write(
    tmp_path,monkeypatch,capsys
):
    monkeypatch.setattr(mod,"SRC",tmp_path/"pilot")
    monkeypatch.setattr(mod,"OUT",tmp_path/"fresh")
    monkeypatch.setattr(mod,"N",12)
    monkeypatch.setattr(mod.s6,"evaluate",lambda trajectory:
                        {"ATE3D":2.45634})
    for name,sigma in mod.pilot.SPECS:
        folder=mod.SRC/name
        folder.mkdir(parents=True)
        np.save(folder/"poses.npy",fake_poses(12))
        (folder/"isj_rot_sigma_manifest.json").write_text(
            json.dumps({
                "spec":mod.pilot.spec_for(name,sigma),
                "metrics":{"ATE3D":2.45634},
            }),
            encoding="utf-8",
        )
    mod.main(["--stage","audit"])
    assert not mod.OUT.exists()
    printed=capsys.readouterr().out
    assert "AUDIT ONLY" in printed
    assert "PENDING" in printed
