from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiments.isj_v8 import diagnose_dev_local_solutions as diag


def _poses():
    arr = np.tile(np.eye(4), (680, 1, 1))
    arr[:, 0, 3] = np.linspace(0, 2, 680)
    return arr


class FakeGtsam:
    class Values:
        def __init__(self):
            self.records = {}
        def insert(self, i, pose):
            self.records[i] = pose

    class Pose3:
        def __init__(self, matrix):
            self.matrix = np.asarray(matrix).copy()


def test_local_probe_pose_values_reconstruction_is_lossless():
    poses = _poses()
    vals = diag.build_values_from_pose_matrices(FakeGtsam, poses)
    assert len(vals.records) == 680
    for i in (0, 100, 679):
        np.testing.assert_array_equal(vals.records[i].matrix, poses[i])


def test_local_probe_rejects_invalid_pose_arrays():
    poses = _poses()
    for corrupted in (
        poses[:679],
        np.where(np.arange(680)[:, None, None] == 1, np.nan, poses),
    ):
        with pytest.raises(ValueError, match=r"invalid SE\(3\)"):
            diag.build_values_from_pose_matrices(FakeGtsam, corrupted)
    bad = poses.copy()
    bad[10, 3, 3] = 0.
    with pytest.raises(ValueError, match="invalid SE"):
        diag.build_values_from_pose_matrices(FakeGtsam, bad)
    bad = poses.copy()
    bad[10, 0, 0] = -1.0
    with pytest.raises(ValueError, match="invalid SE"):
        diag.build_values_from_pose_matrices(FakeGtsam, bad)


def test_local_probe_same_target_objective_comparison_only():
    base = {"objective": 100.0, "stabilized": True}
    cross = {"objective": 90.0, "stabilized": True}
    comp = diag.compare_same_objective(
        base, cross, target_case=diag.CASE_GOOD)
    assert comp["same_target_graph"] is True
    assert comp["cross_seed_objective_is_lower"] is True
    assert comp["distinct_stationary_objectives_detected"] is True
    near = diag.compare_same_objective(
        base, {"objective": 100.000001, "stabilized": True},
        target_case=diag.CASE_GOOD,
    )
    assert near["distinct_stationary_objectives_detected"] is False
    with pytest.raises(ValueError, match="not numerically stable"):
        diag.compare_same_objective(
            base, {"objective": 90., "stabilized": False},
            target_case=diag.CASE_GOOD,
        )


def test_local_probe_hash_gate_protects_frozen_original(monkeypatch, tmp_path):
    original = tmp_path / "original.json"
    original.write_text('{"example":true}', encoding="utf-8")
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    monkeypatch.setattr(diag, "REPORT_PATH", original)
    monkeypatch.setattr(diag, "V8_3_ORIGINAL_SHA256", sha)
    monkeypatch.setattr(
        diag, "audit_report", lambda payload: {"all_stable": True}
    )
    before = original.read_bytes()
    f, summary = diag.load_sealed_source_report()
    assert f["example"] is True
    assert summary["all_stable"] is True
    assert original.read_bytes() == before
    original.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        diag.load_sealed_source_report()


def test_local_probe_preflight_refuses_existing_output(
        monkeypatch, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    monkeypatch.setattr(diag, "OUT", out)
    monkeypatch.setattr(
        diag, "load_sealed_source_report",
        lambda: pytest.fail("source report should not be opened"),
    )
    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        diag.source_preflight()


def test_local_probe_preflight_schedules_exactly_two_frozen_cases(
        monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "OUT", tmp_path / "unused_output")
    mock_sigma = {
        case: np.full((680, 3), index+2., dtype=float)
        for index, case in enumerate(diag.CASES)
    }
    frozen = {
        "gps_degraded_sha256": "D",
        "gps_candidate_sha256": "C",
        "lidar_factors_sha256": "L",
        "camera_factors_sha256": "V",
        "case_sigmas": {
            case: {
                "min_m": float(s.min()),
                "max_m": float(s.max()),
                "mean_xyz_m": np.mean(s, axis=0).tolist(),
            }
            for case, s in mock_sigma.items()
        },
    }
    summary = {"all_stable": True, "metrics": {
        case: {} for case in diag.CASES}}
    monkeypatch.setattr(
        diag, "load_sealed_source_report",
        lambda: (copy.deepcopy(frozen), summary),
    )
    meta = {
        "all_cases_use_same_frame_range": [3180, 3860],
        "heldout_0016_0020_accessed": False,
        **{key: frozen[key] for key in (
            "gps_degraded_sha256", "gps_candidate_sha256",
            "lidar_factors_sha256", "camera_factors_sha256",
        )},
    }
    monkeypatch.setattr(diag, "prepare_cases", lambda: (meta, mock_sigma))
    report, sigmas, _ = diag.source_preflight()
    assert list(sigmas) == list(diag.CASES)
    assert report["case_selection_posthoc_exploratory"] is True
    assert report["reference_influences_optimizer"] is False
    assert report["new_heldout_read"] is False
    assert not (tmp_path / "unused_output").exists()


def test_local_probe_preflight_rejects_modified_sigma(
        monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "OUT", tmp_path / "unused")
    mock_sigma = {case: np.ones((680, 3)) for case in diag.CASES}
    frozen = {
        "gps_degraded_sha256": "D",
        "gps_candidate_sha256": "C",
        "lidar_factors_sha256": "L",
        "camera_factors_sha256": "V",
        "case_sigmas": {
            case: {
                "min_m": 0.5, "max_m": 1.5,
                "mean_xyz_m": [1., 1., 1.],
            }
            for case in diag.CASES
        },
    }
    summary = {"metrics": {case: {} for case in diag.CASES}}
    monkeypatch.setattr(
        diag, "load_sealed_source_report", lambda: (frozen, summary))
    meta = {
        "all_cases_use_same_frame_range": [3180, 3860],
        "heldout_0016_0020_accessed": False,
        **{key: frozen[key] for key in (
            "gps_degraded_sha256", "gps_candidate_sha256",
            "lidar_factors_sha256", "camera_factors_sha256",
        )},
    }
    monkeypatch.setattr(diag, "prepare_cases", lambda: (meta, mock_sigma))
    with pytest.raises(ValueError, match="not frozen"):
        diag.source_preflight()


def test_local_probe_cli_needs_token_before_solver(monkeypatch, capsys):
    executed = []
    monkeypatch.setattr(
        diag, "run_diagnostic",
        lambda: executed.append(True),
    )
    monkeypatch.setattr(
        diag, "source_preflight",
        lambda: ({"source": "0027_DEV"}, {}, {}),
    )
    with pytest.raises(SystemExit):
        diag.main([])
    with pytest.raises(SystemExit):
        diag.main(["--run-probe"])
    with pytest.raises(SystemExit):
        diag.main(["--run-probe", "--token", "NO"])
    assert executed == []
    diag.main(["--preflight"])
    assert "V8_DEV_LOCAL_PROBE_PREFLIGHT_OK" in capsys.readouterr().out
    diag.main(["--run-probe", "--token", diag.TRIGGER_TOKEN])
    assert executed == [True]
