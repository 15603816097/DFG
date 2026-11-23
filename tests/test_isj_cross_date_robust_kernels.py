from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.cross_sequence.run_stage7_robust_baselines import KERNELS
from experiments.isj_supplement.run_cross_date_robust_kernels import (
    SCOPES, MAX_ITER, RTOL, SHIFT_TOL, specs_for, check_cache,
)


def test_frozen_robust_grid_exact_original_thresholds():
    specs = specs_for("both", "both")
    assert len(specs) == 4
    assert len({s["name"] for s in specs}) == 4
    assert {s["kernel"] for s in specs} == {"Huber", "Cauchy"}
    assert {s["scope"] for s in specs} == {"gps", "all"}
    assert all(s["k"] == dict(KERNELS)[s["kernel"]] for s in specs)
    assert specs_for("Huber", "gps")[0]["robust_sensors"] == ["gps"]
    assert specs_for("Cauchy", "all")[0]["robust_sensors"] == [
        "gps", "imu", "camera"
    ]
    assert "lidar" not in [s for sp in specs for s in sp["robust_sensors"]]
    assert json.loads(json.dumps(specs)) == specs
    assert MAX_ITER == 100
    assert RTOL == 1e-7
    assert SHIFT_TOL == 1e-3


def test_invalid_specs_fail():
    with pytest.raises(ValueError):
        specs_for("Hubber", "gps")
    with pytest.raises(ValueError):
        specs_for("both", "nope")
    assert SCOPES["gps"] == ("gps",)


def make_cached(tmp_path, *, stage):
    seq = "2011_09_30_drive_0016_sync"
    spec = specs_for("Huber", "gps")[0]
    name = spec["name"] if stage == "fgo" else "ISJ_CONVERGENCE_" + spec["name"]
    folder = tmp_path / "factor_graph" / name
    folder.mkdir(parents=True)
    np.save(folder / "poses.npy", np.repeat(np.eye(4)[None], 4, axis=0))
    meta = {"sequence": seq, "spec": spec, "stage": stage}
    if stage == "fgo":
        meta.update({
            "metrics": {"ATE3D": 1.2},
            "graph_final_objective": 3.0,
            "graph_initial_objective": 5.0,
        })
    else:
        meta.update({
            "final_metrics": {"ATE3D": 1.2},
            "stationary_under_rule": True,
            "max_extra_passes": 4,
        })
    (folder / "metrics.json").write_text(json.dumps(meta), encoding="utf-8")
    return seq, spec, folder


@pytest.mark.parametrize("stage", ["fgo", "converge"])
def test_cache_reuse_requires_matching_protocol_and_finite_pose(tmp_path, stage):
    seq, spec, folder = make_cached(tmp_path, stage=stage)
    assert check_cache(tmp_path, seq, spec, 4, stage=stage)["sequence"] == seq
    with pytest.raises(ValueError, match="provenance"):
        check_cache(tmp_path, "2011_09_29_drive_0001_sync", spec, 4, stage=stage)
    with pytest.raises(ValueError, match="stage"):
        _check_stage_metadata(folder, tmp_path, seq, spec, stage)
    (folder / "poses.npy").unlink()
    with pytest.raises(RuntimeError, match="Partial cached"):
        check_cache(tmp_path, seq, spec, 4, stage=stage)


def _check_stage_metadata(folder, tmp_path, seq, spec, stage):
    meta_file = folder / "metrics.json"
    info = json.loads(meta_file.read_text())
    info["stage"] = "converge" if stage == "fgo" else "fgo"
    meta_file.write_text(json.dumps(info))
    return check_cache(tmp_path, seq, spec, 4, stage=stage)


def test_no_result_is_safe(tmp_path):
    spec = specs_for("Cauchy", "all")[0]
    assert check_cache(tmp_path, "2011_09_30_drive_0016_sync",
                       spec, 279, stage="fgo") is None
