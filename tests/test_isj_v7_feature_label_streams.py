from __future__ import annotations

import hashlib
import numpy as np
import pytest

from experiments.isj_v7.audit_feature_label_streams import (
    audit_sources, discrepancy,
)
from src.multisensor_reliability.features import gps_features, imu_features


def synth_archives(tmp_path, *, mismatch=False):
    n = 90
    rng = np.random.default_rng(41)
    gps = rng.normal(0, .5, (n, 3)).cumsum(axis=0)
    clean = gps + 0.2
    acc = rng.normal(size=(n, 3))
    gyro = rng.normal(0, .01, size=(n, 3))
    # Input features created from one GPS realization.
    gps_input = gps.copy()
    # Independently generated label-measurement realization.
    gps_label = gps.copy()
    if mismatch:
        gps_label[25:30, 0] += 10.0
    v3_path = tmp_path / "v3.npz"
    degraded_path = tmp_path / "degraded.npz"
    debug_path = tmp_path / "debug.npz"
    np.savez_compressed(
        v3_path,
        gps_features=gps_features(gps_input, dt=.1, window=10),
        imu_features=imu_features(acc, gyro, window=20),
        diagnostic_gps_error_m=np.linalg.norm(gps_label - clean, axis=1),
    )
    np.savez_compressed(
        degraded_path,
        degraded_gps_local=gps_label,
        clean_gps_local=clean,
        degraded_imu_acc=acc,
        degraded_imu_gyro=gyro,
    )
    np.savez_compressed(debug_path, gps_local_corrupted=gps_input)
    return v3_path, degraded_path, debug_path


def test_matched_gps_imu_and_diagnostic_are_consistent(tmp_path):
    src, sensor, debug = synth_archives(tmp_path)
    original = hashlib.sha256(src.read_bytes()).hexdigest()
    result = audit_sources(src, sensor, debug)
    assert original == hashlib.sha256(src.read_bytes()).hexdigest()
    assert result["gps_input_matches_gps_label_measurement"]
    assert result["imu_input_matches_imu_label_measurement"]
    assert result["gps_diagnostic_matches_label_measurement"]
    assert result["debug"]["v2_debug_gps_vs_v3_label_gps"]["allclose"]
    assert result["debug"]["v3_gps_features_vs_v2_debug_gps_features"]["allclose"]
    assert result["approved_for_final_training"] is False
    assert result["status"] == "GPS_MATCH_BUT_LABEL_PROVENANCE_UNAPPROVED"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "debug.npz", "degraded.npz", "v3.npz",
    ]


def test_two_different_gps_realizations_are_detected(tmp_path):
    src, sensor, debug = synth_archives(tmp_path, mismatch=True)
    result = audit_sources(src, sensor, debug)
    assert not result["gps_input_matches_gps_label_measurement"]
    assert result["imu_input_matches_imu_label_measurement"]
    assert result["gps_diagnostic_matches_label_measurement"]
    assert result["status"] == "GPS_REALIZATION_MISMATCH"
    assert result["debug"]["v3_gps_features_vs_v2_debug_gps_features"]["allclose"]
    assert not result["debug"]["v2_debug_gps_vs_v3_label_gps"]["allclose"]
    assert result["comparisons"][
        "v3_gps_features_vs_features_from_label_gps"]["mismatched_rows"] > 0


def test_missing_optional_debug_does_not_change_sensor_result(tmp_path):
    src, sensor, debug = synth_archives(tmp_path)
    debug.unlink()
    result = audit_sources(src, sensor, debug)
    assert not result["debug"]["available"]
    assert result["gps_input_matches_gps_label_measurement"]


def test_invalid_shapes_nonfinite_and_missing_required_data_rejected(tmp_path):
    with pytest.raises(ValueError, match="shape mismatch"):
        discrepancy(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="nonfinite"):
        discrepancy([1., np.nan], [1., 1.])
    src, sensor, debug = synth_archives(tmp_path)
    sensor.unlink()
    with pytest.raises(FileNotFoundError):
        audit_sources(src, sensor, debug)
    assert not sensor.exists()


def test_bad_gps_diagnostic_caught_even_if_features_match(tmp_path):
    src, sensor, debug = synth_archives(tmp_path)
    with np.load(src) as f:
        rows = {k: np.asarray(f[k]).copy() for k in f.files}
    rows["diagnostic_gps_error_m"] += 2.
    np.savez_compressed(src, **rows)
    result = audit_sources(src, sensor, debug)
    assert result["gps_input_matches_gps_label_measurement"]
    assert not result["gps_diagnostic_matches_label_measurement"]
    assert not result["approved_for_final_training"]
