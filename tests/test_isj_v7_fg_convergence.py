from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_v7 import audit_fg_convergence as audit
from src.factor_graph.four_sensor_graph import (
    FourSensorMeasurements, run_four_sensor_factor_graph,
)


def _tiny_case():
    poses = np.tile(np.eye(4, dtype=float), (1, 1, 1))
    return FourSensorMeasurements(
        gps_local=np.array([[0., 0., 0.], [0.1, 0., 0.]], dtype=float),
        imu_gyro=np.zeros((2, 3), dtype=float),
        timestamps=[0., 0.1],
        lidar_between=poses,
        lidar_valid=np.ones(1, dtype=bool),
        lidar_quality=np.ones(1, dtype=float),
        camera_between=poses.copy(),
        camera_valid=np.ones(1, dtype=bool),
        camera_quality=np.ones(1, dtype=float),
    )


def test_graph_only_audit_returns_exact_graph_without_any_writes(tmp_path):
    gtsam = pytest.importorskip("gtsam")
    destination = tmp_path / "must_not_exist"
    m = _tiny_case()
    graph, initial, counts = run_four_sensor_factor_graph(
        m, destination, mode="fixed",
        gps_sigmas_override=np.array([5.0, 6.0]),
        build_graph_only=True,
    )
    assert counts == {"gps": 2, "imu": 1, "lidar": 1, "camera": 1}
    assert graph.size() == 6
    assert initial.size() == 2
    assert np.isfinite(graph.error(initial))
    assert not destination.exists()
    poses = np.tile(np.eye(4, dtype=float), (audit.N, 1, 1))
    values = audit.values_from_poses(gtsam, poses)
    assert values.size() == audit.N


def test_graph_only_no_gps_still_factors_without_writes(tmp_path):
    pytest.importorskip("gtsam")
    m = _tiny_case()
    directory = tmp_path / "must_not_exist"
    graph, values, counts = run_four_sensor_factor_graph(
        m, directory, mode="fixed", use_gps=False,
        build_graph_only=True,
    )
    assert counts["gps"] == 0
    assert graph.size() == 4
    assert values.size() == 2
    assert not directory.exists()


def test_pose_guard_rejects_wrong_shape_nonfinite_rotation_bottom_row():
    correct = np.tile(np.eye(4, dtype=float), (audit.N, 1, 1))
    assert audit.verify_pose_array(correct).shape == (680, 4, 4)
    with pytest.raises(ValueError, match="pose array"):
        audit.verify_pose_array(correct[:-1])
    corrupted = correct.copy()
    corrupted[0, 1, 3] = np.nan
    with pytest.raises(ValueError, match="pose array"):
        audit.verify_pose_array(corrupted)
    corrupted = correct.copy()
    corrupted[0, 3, 0] = 5.
    with pytest.raises(ValueError, match="nonhomogeneous"):
        audit.verify_pose_array(corrupted)
    corrupted = correct.copy()
    corrupted[0, 0, 0] = 2.
    with pytest.raises(ValueError, match="rotations"):
        audit.verify_pose_array(corrupted)


def test_invalid_case_and_seed_fail_before_accessing_archive():
    with pytest.raises(ValueError, match="invalid V7 dev"):
        audit.audit_case("oracle", 43, refine=False)
    with pytest.raises(ValueError, match="invalid V7 dev"):
        audit.audit_case("fixed", 42, refine=False)
    with pytest.raises(ValueError, match="invalid cross-start"):
        audit.audit_case("fixed", 43, refine=False, cross_start="oracle")


def test_objective_audit_writes_nothing_if_saved_output_missing(
        tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUTPUT", tmp_path / "pilot")
    sigma = np.ones(audit.N, dtype=float) * 5.
    with pytest.raises(FileNotFoundError, match="incomplete saved"):
        audit.load_saved_case(
            "fixed", 43, sigma,
            {"gps_measurement_archive_sha256": "a",
             "dataset_archive_sha256": "b",
             "prediction_archive_sha256": None})
    assert not (tmp_path / "pilot").exists()


def test_post_optimization_reference_diagnostics_are_not_independent(
        tmp_path):
    record1, record2 = {}, {}
    poses = np.tile(np.eye(4, dtype=float), (audit.N, 1, 1))
    poses[:, 0, 3] = np.arange(audit.N, dtype=float) * .2
    path = tmp_path / "development_synthetic.npz"
    # Data should be indexed by original raw KITTI 0027 frame index,
    # but the scoring function reads ONLY [3180:3860).
    raw = np.zeros((4544, 3), dtype=np.float64)
    raw[audit.BEGIN:audit.END, 0] = (
        poses[:, 0, 3] + 7.0
    )
    np.savez_compressed(path, clean_gps_local=raw)
    result = {"development_only": True}
    audit._attach_reference_diagnostics(
        result, [(record1, poses), (record2, poses)],
        archive_path=path,
    )
    for r in (record1, record2):
        assert r["reference_independent"] is False
        assert r["reference_used_in_optimization"] is False
        score = r["post_optimization_development_reference_diagnostics"]
        assert score["origin_aligned_oxTS_relative_3d_rmse_m"] < 1e-9
        assert score["independent_ATE"] is False
    assert not (tmp_path / "other-output").exists()


def test_reference_scoring_rejects_bad_interval_or_no_optimization(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, clean_gps_local=np.ones((500, 3)))
    poses = np.tile(np.eye(4), (audit.N, 1, 1))
    with pytest.raises(ValueError, match="reference scoring requires"):
        audit._attach_reference_diagnostics(
            {}, [], archive_path=path,
        )
    with pytest.raises(ValueError, match="development reference"):
        audit._attach_reference_diagnostics(
            {}, [({}, poses)], archive_path=path,
        )
    with pytest.raises(ValueError, match="reference scoring requires"):
        audit.audit_case("fixed", 43, refine=False,
                         score_development_reference=True)
