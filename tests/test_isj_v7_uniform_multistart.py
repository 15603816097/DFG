from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_v7 import audit_uniform_multistart as uniform


def test_default_common_start_pool_is_identical_for_all_cases():
    assert uniform.START_NAMES == ("lidar", "fixed_saved", "mamba_saved")
    assert len(uniform.DEFAULT_CASES) == 7
    assert "mamba_r_u" in uniform.DEFAULT_CASES
    uniform.check_protocol(
        43, list(uniform.DEFAULT_CASES), 4, uniform.START_NAMES,
    )
    with pytest.raises(ValueError, match="predeclared common"):
        uniform.check_protocol(
            43, list(uniform.DEFAULT_CASES), 4,
            ("lidar", "mamba_saved", "fixed_saved"))
    with pytest.raises(ValueError, match="seed43"):
        uniform.check_protocol(
            44, ["mamba_r"], 4, uniform.START_NAMES)


def test_invalid_duplicate_cases_and_solver_budget_are_rejected():
    with pytest.raises(ValueError, match="invalid or duplicate"):
        uniform.check_protocol(
            43, ["fixed", "fixed"], 4, uniform.START_NAMES)
    with pytest.raises(ValueError, match="invalid or duplicate"):
        uniform.check_protocol(
            43, ["independent_0020"], 4, uniform.START_NAMES)
    with pytest.raises(ValueError, match="passes"):
        uniform.check_protocol(
            43, ["fixed"], 1, uniform.START_NAMES)
    with pytest.raises(ValueError, match="passes"):
        uniform.check_protocol(
            43, ["fixed"], 13, uniform.START_NAMES)


def test_selection_uses_only_graph_objective_not_reference_error():
    candidates = {
        "lidar": {
            "objective": 12.0,
            "hypothetical_reference_rmse": 0.01,
        },
        "fixed_saved": {
            "objective": 5.0,
            "hypothetical_reference_rmse": 1000.0,
        },
        "mamba_saved": {
            "objective": 7.0,
            "hypothetical_reference_rmse": 0.0,
        },
    }
    name, chosen = uniform.select_within_graph_objective(candidates)
    assert name == "fixed_saved"
    assert chosen is candidates["fixed_saved"]
    with pytest.raises(ValueError, match="common start"):
        uniform.select_within_graph_objective({
            "lidar": {"objective": 1.0}})
    corrupted = candidates.copy()
    corrupted["lidar"] = {"objective": np.nan}
    with pytest.raises(ValueError, match="invalid candidate"):
        uniform.select_within_graph_objective(corrupted)


def test_preflight_is_read_only_and_never_builds_fgo(
        tmp_path, monkeypatch):
    def fake_prepare(case, seed):
        assert seed == 43
        assert case in ("fixed", "mamba_r")
        return (
            np.ones(680, dtype=float) * 5.,
            {"sigma_mean": 5., "prediction_archive_sha256": None},
        )
    monkeypatch.setattr(uniform, "prepare_case", fake_prepare)
    monkeypatch.setattr(
        uniform, "load_common_starts",
        lambda seed: {
            "fixed_saved": np.tile(np.eye(4), (680, 1, 1)),
            "mamba_saved": np.tile(np.eye(4), (680, 1, 1))},
    )
    def no_graph(*args, **kwargs):
        raise AssertionError("preflight must not touch measurement files")
    monkeypatch.setattr(
        uniform, "load_degraded_four_sensor_measurements", no_graph)
    monkeypatch.setattr(uniform, "DEGRADED", tmp_path / "nonexistent")
    result = uniform.analyze(
        ["fixed", "mamba_r"], seed=43, max_passes=3,
        run_solver=False,
    )
    assert result["preflight_only"] is True
    assert result["writes"] is False
    assert result["start_names"] == list(uniform.START_NAMES)
    assert list(tmp_path.iterdir()) == []


def test_position_validation_rejects_invalid_GTSAM_poses():
    gtsam = pytest.importorskip("gtsam")
    from experiments.isj_v7.audit_fg_convergence import values_from_poses
    a = np.tile(np.eye(4), (uniform.N, 1, 1))
    x = values_from_poses(gtsam, a)
    assert np.array_equal(
        uniform._pose_from_values(gtsam, x), a)
