from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.isj_supplement.run_cross_date_residual_baseline import (
    select_specs, saved_case,
)


def _fake_case(tmp_path, spec, n=4):
    out = tmp_path
    folder = out / "factor_graph" / spec["name"]
    folder.mkdir(parents=True)
    poses = np.repeat(np.eye(4)[None, :, :], n, axis=0)
    np.save(folder / "poses.npy", poses)
    meta = {
        "sequence": "2011_09_30_drive_0016_sync",
        "case": spec["name"],
        "selected_sensors": list(spec["sensors"]),
        "warm_start": "factor_graph/FINAL_S0_fixed/poses.npy",
        "protocol": "exploratory_cross_date_one_step_residual_adaptive",
        "graph_initial_objective": 12.4,
        "graph_final_objective": 6.2,
        "metrics": {"ATE3D": 1.5, "ATE2D": 1.4, "ZRMSE": 0.2},
    }
    (folder / "metrics.json").write_text(json.dumps(meta))
    return out, folder, meta


def test_two_baselines_and_json_stable_convergence_metadata():
    specs = select_specs("both")
    assert len(specs) == 2
    assert len({s["name"] for s in specs}) == 2
    assert all(s["type"] == "adaptive" for s in specs)
    assert json.loads(json.dumps(specs)) == specs
    assert select_specs("gps")[0]["sensors"] == ["gps"]
    assert select_specs("all")[0]["sensors"] == ["gps", "imu", "camera"]
    with pytest.raises(ValueError):
        select_specs("nonexistent")


def test_absent_case_is_safe_to_compute(tmp_path):
    assert saved_case(tmp_path, "2011_09_30_drive_0016_sync",
                      select_specs("gps")[0], 279) is None


def test_cache_load_requires_matching_source_and_paired_files(tmp_path):
    spec = select_specs("gps")[0]
    out, folder, meta = _fake_case(tmp_path, spec)
    result = saved_case(out, meta["sequence"], spec, 4)
    assert result["metrics"]["ATE3D"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="metadata mismatch"):
        saved_case(out, "2011_09_29_drive_0001_sync", spec, 4)
    (folder / "poses.npy").unlink()
    with pytest.raises(RuntimeError, match="Incomplete result"):
        saved_case(out, meta["sequence"], spec, 4)


def test_invalid_saved_objective_and_pose_rejected(tmp_path):
    spec = select_specs("all")[0]
    out, folder, meta = _fake_case(tmp_path, spec)
    meta["graph_final_objective"] = float("nan")
    (folder / "metrics.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="invalid graph_final_objective"):
        saved_case(out, meta["sequence"], spec, 4)
    meta["graph_final_objective"] = 4.3
    (folder / "metrics.json").write_text(json.dumps(meta))
    np.save(folder / "poses.npy", np.ones((3, 4, 4)))
    with pytest.raises(ValueError, match="Invalid saved poses"):
        saved_case(out, meta["sequence"], spec, 4)
