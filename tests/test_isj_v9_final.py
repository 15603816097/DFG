"""Synthetic-only V9 final runner tests; no new drive measurements."""
from pathlib import Path
import numpy as np
import pytest

from experiments.isj_v8.run_v9_final import (
    METHODS, metrics, PRIMARY, identity_array, checked_case, write_new, sha,
)
from experiments.isj_v8.evaluate_c8_refined_dev import metrics as dev_metrics


@pytest.mark.parametrize("n",[279,339,447,474,680,1104])
def test_variable_length_reference_metrics(n):
    x=np.arange(n,dtype=float)
    clean=np.column_stack([x, .02*x, .001*x])+np.array([10.,20.,30.])
    t=clean-clean[0]+np.array([3.,4.,12.])
    result=metrics(t,clean)
    assert result[PRIMARY]==pytest.approx(13.)
    assert result["origin_aligned_oxTS_relative_2d_rmse_m"]==pytest.approx(5.)
    assert result["p95_3d_m"]==pytest.approx(13.)
    assert result["vertical_rmse_m"]==pytest.approx(12.)
    assert result["relative_displacement_1_frame_rmse_m"]==pytest.approx(0.,abs=1e-12)
    assert result["relative_displacement_10_frame_rmse_m"]==pytest.approx(0.,abs=1e-12)
    if n==680:
        old=dev_metrics(t,clean)
        for key,val in result.items():
            assert val==pytest.approx(old[key])


def test_matrix_and_array_identity():
    assert len(METHODS)==len(set(METHODS))==20
    assert len([m for m in METHODS if m.endswith("__calibrated")])==9
    assert len([m for m in METHODS if m.endswith("__original")])==9
    x=np.arange(6,dtype=float).reshape(2,3)
    assert identity_array(x)==identity_array(x.copy())
    assert identity_array(x)!=identity_array(x.reshape(3,2))
    assert identity_array(x)!=identity_array(x.astype(np.float32))


def test_failed_case_kept_without_poses(tmp_path):
    expected={"frames":8,"method":"axis_fixed"}
    rec=dict(input=expected,status="SOLVER_ERROR",poses_sha256=None,error="synthetic")
    write_new(tmp_path/"case.json",rec)
    assert checked_case(tmp_path,expected)==rec
    np.save(tmp_path/"poses.npy",np.zeros((8,4,4)))
    with pytest.raises(ValueError,match="failed case contains poses"):
        checked_case(tmp_path,expected)


def test_pose_tampering_and_not_stable(tmp_path):
    expected={"frames":8,"method":"axis_fixed"}
    p=tmp_path/"poses.npy"
    np.save(p,np.repeat(np.eye(4)[None],8,axis=0))
    rec=dict(input=expected,status="NOT_STABLE",poses_sha256=sha(p),
             health={"derivative_audit":{"passed":True},"last_step_stable":False})
    write_new(tmp_path/"case.json",rec)
    assert checked_case(tmp_path,expected)==rec
    np.save(p,np.zeros((8,4,4)))
    with pytest.raises(ValueError,match="pose hash drift"):
        checked_case(tmp_path,expected)
