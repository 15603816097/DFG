from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.isj_v8 import run_dev_solver_health_replay as r


class FakePose:
    def __init__(self, arr):
        self.matrix = np.asarray(arr).copy()


class FakeValues:
    def __init__(self, poses=None):
        self.records = {}
        self.poses = poses
    def insert(self, i, pose):
        self.records[i] = pose


class FakeGtsam:
    Values = FakeValues
    Pose3 = FakePose


def _poses():
    pos = np.tile(np.eye(4), (680, 1, 1))
    pos[:, 0, 3] = np.linspace(0, 20, 680)
    return pos


def test_v8_3_3_common_reference_free_seeds_from_degraded_gps(monkeypatch):
    original = FakeValues(_poses())
    monkeypatch.setattr(
        r, "_positions", lambda _g, val, _n: val.poses.copy()
        if val.poses is not None else
        np.stack([val.records[i].matrix for i in range(680)])
    )
    gps = np.zeros((680, 3))
    gps[:, 1] = np.linspace(0, 10, 680)
    seeds = r.common_initializations(FakeGtsam, original, gps)
    assert list(seeds) == list(r.SEED_NAMES)
    assert seeds["common_lidar_odometry"] is original
    second = seeds["degraded_gps_translation_lidar_rotation"]
    frames = np.stack([second.records[i].matrix for i in range(680)])
    np.testing.assert_allclose(frames[:, :3, 3], gps)
    np.testing.assert_allclose(
        frames[:, :3, :3], original.poses[:, :3, :3])


def test_v8_3_3_common_seeds_reject_nonorigin_or_wrong_framecount(monkeypatch):
    monkeypatch.setattr(r, "_positions", lambda g, v, n: _poses())
    bad = np.ones((680, 3))
    with pytest.raises(ValueError, match="rebased"):
        r.common_initializations(FakeGtsam, FakeValues(_poses()), bad)
    with pytest.raises(ValueError, match="rebased"):
        r.common_initializations(FakeGtsam, FakeValues(_poses()), np.zeros((679, 3)))


def _mock_preflight_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "OUT", tmp_path / "uncreated")
    stationarity = tmp_path / "frozen_stationarity.json"
    stationarity.write_text(json.dumps({
        "source_v8_3_1_sha256": r.EXPECTED_PREVIOUS_SHA256,
        "heldout_0016_0020_accessed": False,
    }), encoding="utf-8")
    monkeypatch.setattr(r, "STATIONARITY_PATH", stationarity)
    monkeypatch.setattr(
        r, "EXPECTED_V8_3_STATIONARITY_SHA", r.file_sha256(stationarity))
    sigmas = {
        case: np.full((680, 3), 2., dtype=float) for case in r.CASES
    }
    frozen = {
        "case_sigmas": {
            case: {
                "min_m": 2., "max_m": 2., "mean_xyz_m": [2., 2., 2.]
            } for case in r.CASES
        }
    }
    fresh = {
        "all_cases_use_same_frame_range": [3180, 3860],
        "heldout_0016_0020_accessed": False,
    }
    for key in (
        "gps_degraded_sha256", "gps_candidate_sha256",
        "lidar_factors_sha256", "camera_factors_sha256",
    ):
        frozen[key] = fresh[key] = "FROZEN"
    monkeypatch.setattr(r, "load_sealed_source_report",
                        lambda: (frozen, {"all_stable": True}))
    monkeypatch.setattr(r, "prepare_cases", lambda: (fresh, sigmas))
    # Normally production file_sha256 of frozen V8.3 original is
    # invoked here. Create a tiny source marker instead without
    # altering the actual user results.
    repo_root = tmp_path / "root"
    parent = (repo_root / "results/isj_v8/dev_0027_feedforward_fgo_v1")
    parent.mkdir(parents=True)
    (parent / "v8_dev_feedforward_fgo_results.json").write_bytes(b"OLD")
    monkeypatch.setattr(r, "ROOT", repo_root)
    return sigmas


def test_v8_3_3_preflight_freezes_case_mapping_without_writes(
        monkeypatch, tmp_path):
    sigmas = _mock_preflight_sources(monkeypatch, tmp_path)
    meta, recovered = r.source_preflight()
    assert tuple(recovered) == r.CASES
    assert list(meta["cases"]) == list(r.CASES)
    assert meta["initialization_names"] == list(r.SEED_NAMES)
    assert meta["initialization_selection_uses_OXTS"] is False
    assert meta["candidate_selection"].startswith("min_same_target")
    assert not r.OUT.exists()


def test_v8_3_3_preflight_refuses_changed_frozen_covariance(
        monkeypatch, tmp_path):
    sigmas = _mock_preflight_sources(monkeypatch, tmp_path)
    sigmas[r.CASES[0]][0, 0] = 3.0
    with pytest.raises(ValueError, match="covariance changed"):
        r.source_preflight()


def test_v8_3_3_preflight_refuses_existing_output_before_reads(
        monkeypatch, tmp_path):
    folder = tmp_path / "existing"
    folder.mkdir()
    monkeypatch.setattr(r, "OUT", folder)
    monkeypatch.setattr(r, "load_sealed_source_report",
                        lambda: pytest.fail("must not read sources"))
    with pytest.raises(FileExistsError, match="refuse overwrite"):
        r.source_preflight()


def test_v8_3_3_cli_disallows_unapproved_optimization(monkeypatch, capsys):
    events = []
    monkeypatch.setattr(r, "source_preflight", lambda: (
        {"development_only": True}, {r.CASES[0]: np.ones((680, 3))},
    ))
    monkeypatch.setattr(r, "run_replay", lambda: events.append("SOLVE"))
    with pytest.raises(SystemExit):
        r.main([])
    with pytest.raises(SystemExit):
        r.main(["--run-replay"])
    with pytest.raises(SystemExit):
        r.main(["--run-replay", "--token", "SEALED_0016_0020"])
    r.main(["--preflight"])
    assert "V8_3_3_PREFLIGHT_OK" in capsys.readouterr().out
    assert not events
    r.main(["--run-replay", "--token", r.TOKEN])
    assert events == ["SOLVE"]


def test_v8_3_3_full_runner_selects_only_candidate_with_no_truth_selection(
        monkeypatch, tmp_path):
    out = tmp_path / "new_results"
    monkeypatch.setattr(r, "OUT", out)
    monkeypatch.setattr(r, "METADATA", out / "v8_dev_solver_health_results.json")
    monkeypatch.setattr(r, "POSE_ARCHIVE", out / "all_solver_poses.npz")
    source = tmp_path / "degraded_sensor_data.npz"
    np.savez(source, clean_gps_local=np.zeros((4544, 3)))
    monkeypatch.setattr(r, "SOURCE", source)
    monkeypatch.setitem(sys.modules, "gtsam", FakeGtsam)
    # Reuse all 16 sources, no real GTSAM solve in this unit test.
    sigmas = {
        case: np.full((680, 3), i+2., dtype=float)
        for i, case in enumerate(r.CASES)
    }
    monkeypatch.setattr(r, "source_preflight", lambda: (
        {"status": "PREFLIGHT_ONLY", "source_v8_3_sha256": "SOURCE_HASH"},
        sigmas,
    ))
    monkeypatch.setattr(r, "load_degraded_four_sensor_measurements",
                        lambda _: object())
    class Sub:
        gps_local = np.zeros((680, 3))
    monkeypatch.setattr(r, "subgraph_measurements", lambda raw, **kw: Sub())
    class Init:
        poses = _poses()
    def create_graph(data, outpath, **kw):
        assert kw["build_graph_only"] is True
        assert kw["mode"] == "fixed"
        return object(), Init(), r.FACTORS.copy()
    monkeypatch.setattr(r, "run_four_sensor_factor_graph", create_graph)
    monkeypatch.setattr(r, "_positions", lambda _g, v, n: v.poses.copy())
    monkeypatch.setattr(r, "common_initializations",
                        lambda g, init, gps: {
                            r.SEED_NAMES[0]: "LIDAR",
                            r.SEED_NAMES[1]: "GPS",
                        })
    call = []
    def solve(graph, values, n):
        call.append(values)
        status = ("CONVERGED_CANDIDATE" if values == "LIDAR"
                  else "STALLED_NO_PROGRESS")
        return {
            "status": status, "poses": _poses(),
            "objective": 10. if values == "LIDAR" else 1.,
            "initial_objective": 100.,
            "initial_gradient_l2": 100.,
            "final_gradient_l2": 2. if values == "LIDAR" else 200.,
            "passes": [{"status": status}],
        }
    monkeypatch.setattr(r, "solve_v8_health", solve)
    monkeypatch.setattr(r, "reference_diagnostics",
                        lambda poses, ref: {"3D_DEV_ONLY": 1.0})
    report = r.run_replay()
    assert len(call) == 32
    assert report["candidate_case_count"] == 16
    assert report["cases_without_qualified_candidate"] == []
    assert all(
        report["case_results"][c]["selection"]["selected_seed"] ==
        r.SEED_NAMES[0] for c in r.CASES
    )
    assert report["evaluation_not_independent"] is True
    assert out.is_dir()
    assert r.POSE_ARCHIVE.is_file()
    assert r.METADATA.is_file()
    with np.load(r.POSE_ARCHIVE, allow_pickle=False) as data:
        assert len(data.files) == 34
