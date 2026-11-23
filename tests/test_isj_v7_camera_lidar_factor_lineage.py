from __future__ import annotations

import hashlib
import numpy as np
import pytest

from experiments.isj_v7.audit_camera_lidar_factor_lineage import (
    audit_factor_label_lineage,
    compare_special,
    derive_proxy,
)
from src.reliability.oracle_factor_reliability import OracleReliabilityConfig


def _make_fixture(tmp_path, *, corrupt_saved=False):
    n = 24
    reference = np.repeat(np.eye(4)[None], n, axis=0)
    reference[:, 0, 3] = np.linspace(0, 2, n)
    reference_path = tmp_path / "poses.npy"
    np.save(reference_path, reference)
    lidar_between = np.repeat(np.eye(4)[None], n - 1, axis=0)
    camera_between = lidar_between.copy()
    step = reference[1, 0, 3] - reference[0, 0, 3]
    lidar_between[:, 0, 3] = step + .09
    camera_between[:, 0, 3] = step + .13
    lidar_valid = np.ones(n - 1, dtype=bool)
    camera_valid = np.ones(n - 1, dtype=bool)
    lidar_valid[4] = False
    camera_valid[7] = False
    # V3 builder ignores non-converged LiDAR even when factor transform finite.
    lidar_factor = tmp_path / "lidar.npz"
    camera_factor = tmp_path / "camera.npz"
    np.savez_compressed(
        lidar_factor, body_between=lidar_between,
        converged=lidar_valid.astype(np.int32))
    np.savez_compressed(
        camera_factor, between_measurements=camera_between,
        valid=camera_valid)
    cfg = OracleReliabilityConfig()
    lidar_label = derive_proxy(
        lidar_between, lidar_valid, reference,
        translation_scale_m=cfg.lidar_translation_scale_m,
        rotation_scale_deg=cfg.lidar_rotation_scale_deg)
    camera_label = derive_proxy(
        camera_between, camera_valid, reference,
        translation_scale_m=cfg.camera_translation_scale_m,
        rotation_scale_deg=cfg.camera_rotation_scale_deg)
    archive = {
        "lidar_features": np.zeros((n, 19), dtype=np.float32),
        "camera_features": np.zeros((n, 16), dtype=np.float32),
    }
    for name, derived in (
        ("lidar", lidar_label), ("camera", camera_label)
    ):
        archive[f"{name}_current_factor_reliability"] = (
            derived["reliability"].astype(np.float32))
        archive[f"diagnostic_{name}_translation_error_m"] = (
            derived["translation_error"])
        archive[f"diagnostic_{name}_rotation_error_deg"] = (
            derived["rotation_error_deg"])
    if corrupt_saved:
        archive["camera_current_factor_reliability"][5] = .01
    v3 = tmp_path / "v3.npz"
    np.savez_compressed(v3, **archive)
    return v3, lidar_factor, camera_factor, reference_path


def test_saved_factor_labels_are_reconstructed_without_writes(tmp_path):
    paths = _make_fixture(tmp_path)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    report = audit_factor_label_lineage(*paths)
    assert report["frames"] == 24
    assert report["saved_factor_label_lineage_matches"]
    assert report["approved_for_final_training"] is False
    assert report["independent_evaluation_established"] is False
    assert report["modalities"]["lidar"]["invalid_pairs"] == 1
    assert report["modalities"]["camera"]["invalid_pairs"] == 1
    for name in ("lidar", "camera"):
        assert report["modalities"][name]["all_saved_proxy_fields_match"]
        assert not report["modalities"][name][
            "feature_input_vs_factor_raw_measurement_verified"]
    assert before == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert len(list(tmp_path.iterdir())) == 4


def test_mismatched_saved_camera_label_is_detected(tmp_path):
    paths = _make_fixture(tmp_path, corrupt_saved=True)
    result = audit_factor_label_lineage(*paths)
    assert not result["saved_factor_label_lineage_matches"]
    assert result["modalities"]["lidar"]["all_saved_proxy_fields_match"]
    assert not result["modalities"]["camera"]["all_saved_proxy_fields_match"]
    assert result["modalities"]["camera"]["source_factor_vs_v3_label"][
        "current_reliability"]["mismatched_frames"] == 1


def test_invalid_factor_produces_inf_diagnostics_but_minimum_reliability():
    n = 6
    pose = np.repeat(np.eye(4)[None], n, axis=0)
    between = np.repeat(np.eye(4)[None], n - 1, axis=0)
    valid = np.ones(n - 1, bool)
    valid[2] = False
    derived = derive_proxy(
        between, valid, pose,
        translation_scale_m=.5, rotation_scale_deg=1.)
    assert derived["translation_error"][3] == np.inf
    assert derived["rotation_error_deg"][3] == np.inf
    assert derived["reliability"][3] == pytest.approx(.01)
    assert derived["reliability"][0] == pytest.approx(1.0)
    assert compare_special(
        [0, float("inf")], [0, float("inf")])["allclose"]
    assert not compare_special(
        [0, float("inf")], [0, float("-inf")])["allclose"]


def test_archive_shapes_and_missing_path_fail_closed(tmp_path):
    paths = _make_fixture(tmp_path)
    v3, lidar, camera, reference = paths
    with np.load(lidar) as src:
        contents = {k: np.asarray(src[k]).copy() for k in src.files}
    contents["converged"] = contents["converged"][:-1]
    np.savez_compressed(lidar, **contents)
    with pytest.raises(ValueError, match="shapes"):
        audit_factor_label_lineage(*paths)
    missing = tmp_path / "none.npz"
    with pytest.raises(FileNotFoundError):
        audit_factor_label_lineage(v3, missing, camera, reference)
    assert not missing.exists()


def test_heldout_sequence_path_is_rejected(tmp_path):
    paths = _make_fixture(tmp_path)
    reserved = tmp_path / "0020"
    reserved.mkdir()
    with pytest.raises(ValueError, match="reserved"):
        audit_factor_label_lineage(
            paths[0], paths[1], reserved / "missing.npz", paths[3])
