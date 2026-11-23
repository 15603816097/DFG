from __future__ import annotations

import json
from pathlib import Path
import types

import numpy as np
import pytest

from experiments.isj_v8 import diagnose_dev_stationarity as st


class FakeGraph:
    def __init__(self, objective=22.0, gradient=7.0):
        self.objective = objective
        self.gradient = gradient

    def error(self, state):
        return self.objective

    def linearize(self, values):
        g = self.gradient
        return types.SimpleNamespace(
            gradientAtZero=lambda: types.SimpleNamespace(norm=lambda: g)
        )


class FakeValues:
    def __init__(self):
        self.data = {}

    def insert(self, i, pose):
        self.data[i] = pose


class FakePose3:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix).copy()


FAKE_GTSAM = types.SimpleNamespace(Values=FakeValues, Pose3=FakePose3)


def _solution():
    return {
        "poses": np.tile(np.eye(4), (680, 1, 1)),
        "objective": 22.,
        "stabilized": True,
        "passes": [{"stabilized": True,
                    "position_shift_rmse_m": 0.,
                    "relative_objective_reduction": 0.}],
    }


def test_stationarity_gtsam_linearized_gradient_norm_is_reported():
    g = FakeGraph(gradient=31.0)
    assert st.gradient_norm(g, FakeValues()) == 31.
    assert st.gradient_norm(FakeGraph(gradient=0.), FakeValues()) == 0.


def test_stationarity_rejects_unsupported_gradient():
    g = types.SimpleNamespace(linearize=lambda vals: object())
    with pytest.raises(RuntimeError, match="unsupported"):
        st.gradient_norm(g, FakeValues())


def test_stationarity_rejects_nonfinite_gradient():
    with pytest.raises(RuntimeError, match="invalid"):
        st.gradient_norm(FakeGraph(gradient=float("nan")), FakeValues())


def test_stationarity_replays_exact_objective_and_requires_stability():
    report = st.inspect_result(FAKE_GTSAM, FakeGraph(22., 14.), _solution())
    assert report["objective"] == pytest.approx(22.)
    assert report["gradient_l2"] == pytest.approx(14.)
    assert report["gradient_l2_per_sqrt_tangent_dimension"] == pytest.approx(
        14. / np.sqrt(680 * 6))
    bad = _solution()
    bad["objective"] = 23.
    with pytest.raises(ValueError, match="reconstructed terminal"):
        st.inspect_result(FAKE_GTSAM, FakeGraph(), bad)
    bad = _solution()
    bad["stabilized"] = False
    with pytest.raises(ValueError, match="reconstructed terminal"):
        st.inspect_result(FAKE_GTSAM, FakeGraph(), bad)


def test_stationarity_source_gates_existing_destination_before_file_read(
        monkeypatch, tmp_path):
    path = tmp_path / "old"
    path.mkdir()
    monkeypatch.setattr(st, "OUT", path)
    monkeypatch.setattr(
        st, "load_sealed_source_report",
        lambda: pytest.fail("source should not be opened"),
    )
    with pytest.raises(FileExistsError, match="destination exists"):
        st.validate_existing_probes()


def test_stationarity_protects_v8_3_1_sha256(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "OUT", tmp_path / "new")
    p = tmp_path / "old_probe.json"
    p.write_text('{"invalid": true}', encoding="utf-8")
    monkeypatch.setattr(st, "PREVIOUS_REPORT", p)
    monkeypatch.setattr(
        st, "load_sealed_source_report", lambda: ({}, {}),
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        st.validate_existing_probes()


def test_stationarity_preflight_fails_on_invalid_axis_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "OUT", tmp_path / "new")
    p = tmp_path / "prior.json"
    prior = {
        "status": "COMPLETE_0027_DEV_EXPLORATORY_LOCAL_PROBE",
        "source_v8_3_json_sha256": st.V8_3_ORIGINAL_SHA256,
        "new_heldout_read": False,
        "solves": {name: {} for name in st.CASES},
    }
    p.write_text(json.dumps(prior), encoding="utf-8")
    monkeypatch.setattr(st, "PREVIOUS_REPORT", p)
    monkeypatch.setattr(st, "EXPECTED_PREVIOUS_SHA256", st.sha(p))
    frozen = {
        "case_sigmas": {
            c: {"min_m": 1.0, "max_m": 1.0,
                "mean_xyz_m": [1., 1., 1.]} for c in st.CASES},
    }
    fresh = {
        "all_cases_use_same_frame_range": [3180, 3860],
        "heldout_0016_0020_accessed": False,
    }
    for key in (
        "gps_degraded_sha256", "gps_candidate_sha256",
        "lidar_factors_sha256", "camera_factors_sha256",
    ):
        frozen[key] = fresh[key] = "SHA"
    monkeypatch.setattr(st, "load_sealed_source_report",
                        lambda: (frozen, {}))
    monkeypatch.setattr(st, "prepare_cases", lambda: (
        fresh, {c: np.ones((680, 3))*2. for c in st.CASES}
    ))
    with pytest.raises(ValueError, match="sigma mapping changed"):
        st.validate_existing_probes()


def test_stationarity_cli_rejects_unsafe_actions(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(
        st, "validate_existing_probes",
        lambda: ({}, {case: np.ones((680, 3)) for case in st.CASES}),
    )
    monkeypatch.setattr(
        st, "run_diagnostic", lambda: called.append(True),
    )
    with pytest.raises(SystemExit):
        st.main([])
    with pytest.raises(SystemExit):
        st.main(["--run-diagnostic"])
    with pytest.raises(SystemExit):
        st.main(["--preflight", "--token", st.TOKEN])
    assert not called
    st.main(["--preflight"])
    assert "V8_DEV_STATIONARITY_PREFLIGHT_OK" in capsys.readouterr().out
    st.main(["--run-diagnostic", "--token", st.TOKEN])
    assert called == [True]
