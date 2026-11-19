from __future__ import annotations

import hashlib
import numpy as np
import pytest

from experiments.isj_v7.audit_gps_canonical_dryrun import (
    preview_gps_rebuild,
)
from src.multisensor_reliability.features import gps_features
from src.reliability.oracle_factor_reliability import OracleReliabilityConfig


def make_data(tmp_path, *, mismatch=True, wrong_label=False):
    n = 180
    rng = np.random.default_rng(1234)
    z = rng.normal(0, .3, (n, 3)).cumsum(axis=0)
    clean = z - np.sin(np.arange(n)[:, None] / 13.0)
    err = np.linalg.norm(z - clean, axis=1)
    config = OracleReliabilityConfig()
    y = np.clip(np.exp(-np.square(err / config.gps_position_scale_m)),
                config.minimum_reliability,
                config.maximum_reliability).astype(np.float32)
    if wrong_label:
        y[25] = 0.01
    old_z = z.copy()
    if mismatch:
        old_z[60:68, 0] += 20
    p1 = tmp_path / "v3.npz"
    p2 = tmp_path / "degraded.npz"
    np.savez_compressed(
        p1,
        gps_features=gps_features(old_z, dt=.1, window=10),
        gps_current_factor_reliability=y,
        diagnostic_gps_error_m=err,
    )
    np.savez_compressed(
        p2,
        degraded_gps_local=z,
        clean_gps_local=clean,
    )
    return p1, p2


def test_inmemory_same_realization_can_be_reconstructed_without_writes(tmp_path):
    a, b = make_data(tmp_path)
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (a, b)
    }
    result = preview_gps_rebuild(a, b, window=8, horizon=3)
    assert result["gps_same_realization_candidate_consistent"]
    assert result["saved_gps_error_vs_same_stream_oracle_error"]["allclose"]
    assert result["saved_gps_reliability_vs_same_stream_oracle_mapping"]["allclose"]
    assert not result["old_v3_features_vs_canonical"]["allclose"]
    assert result["old_v3_features_vs_canonical"]["mismatched_rows"] > 0
    assert result["head_offsets"] == [0, 1, 2]
    assert result["approved_for_final_training"] is False
    for part in ("train", "val", "test"):
        p = result["samples"][part]
        assert p["count"] > 0
        hi = {
            "train": result["split_train_end"],
            "val": result["split_val_end"],
            "test": 180,
        }[part]
        assert p["last_target"] < hi
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (a, b)
    }
    assert len(list(tmp_path.iterdir())) == 2


def test_original_features_match_when_same_realization(tmp_path):
    a, b = make_data(tmp_path, mismatch=False)
    result = preview_gps_rebuild(a, b, window=8, horizon=3)
    assert result["old_v3_features_vs_canonical"]["allclose"]


def test_saved_label_corruption_is_not_approved(tmp_path):
    a, b = make_data(tmp_path, wrong_label=True)
    report = preview_gps_rebuild(a, b, window=8, horizon=3)
    assert not report["gps_same_realization_candidate_consistent"]
    assert not report["saved_gps_reliability_vs_same_stream_oracle_mapping"]["allclose"]
    assert not report["approved_for_final_training"]


def test_bad_archive_shape_rejected(tmp_path):
    a, b = make_data(tmp_path)
    with np.load(b) as src:
        payload = {k: np.asarray(src[k]).copy() for k in src.files}
    payload["degraded_gps_local"] = payload["degraded_gps_local"][:-1]
    np.savez_compressed(b, **payload)
    with pytest.raises(ValueError, match="inconsistent"):
        preview_gps_rebuild(a, b, window=8, horizon=3)


def test_missing_archive_does_not_trigger_generation(tmp_path):
    a, b = make_data(tmp_path)
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        preview_gps_rebuild(a, missing)
    assert not missing.exists()
