from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.isj_v7.build_dev_gps_candidate import (
    prepare_dev_candidate, write_candidate,
)
from src.multisensor_reliability.features import gps_features
from src.reliability.oracle_factor_reliability import OracleReliabilityConfig


def fixtures(tmp_path, *, bad_label=False):
    n = 240
    rng = np.random.default_rng(20261119)
    degraded = rng.normal(0, 0.25, (n, 3)).cumsum(axis=0)
    clean = degraded - np.column_stack([
        np.sin(np.arange(n) / 23),
        np.cos(np.arange(n) / 17),
        0.1 * np.sin(np.arange(n) / 7),
    ])
    error = np.linalg.norm(degraded - clean, axis=1)
    cfg = OracleReliabilityConfig()
    y = np.clip(np.exp(-np.square(error / cfg.gps_position_scale_m)),
                cfg.minimum_reliability, cfg.maximum_reliability
                ).astype(np.float32)
    if bad_label:
        y[31] = 0.01

    # The original 0027 V3 archive has *different* GPS feature realization.
    wrong_input = degraded.copy()
    wrong_input[45:57, 0] += 12
    saved = tmp_path / "v3.npz"
    measurement = tmp_path / "degraded.npz"
    np.savez_compressed(
        saved, gps_features=gps_features(wrong_input, dt=0.1, window=10),
        gps_current_factor_reliability=y,
        diagnostic_gps_error_m=error,
    )
    np.savez_compressed(
        measurement, degraded_gps_local=degraded,
        clean_gps_local=clean,
    )
    return saved, measurement


def test_gps_candidate_train_val_prefix_only(tmp_path):
    v3, measured = fixtures(tmp_path)
    data, info = prepare_dev_candidate(v3, measured, window=8, horizon=3)
    assert info["development_only"] is True
    assert info["approved_for_final_training"] is False
    assert info["omitted_raw_clean_gps"] is True
    assert info["source_frames"] == 240
    assert info["exported_feature_frames"] == int(.85 * 240)
    assert info["omitted_internal_test_range"] == [204, 240]
    assert set(data) == {
        "gps_features", "train_anchors", "train_targets",
        "val_anchors", "val_targets", "train_median", "train_iqr",
    }
    assert data["gps_features"].shape == (204, 21)
    assert data["train_targets"].shape[1] == 3
    assert data["val_targets"].shape[1] == 3
    assert len(data["train_anchors"]) == info["train_anchor_count"]
    assert len(data["val_anchors"]) == info["val_anchor_count"]
    assert data["train_anchors"][-1] + 2 < 168
    assert data["val_anchors"][0] >= 168
    assert data["val_anchors"][-1] + 2 < 204
    assert info["forecast_heads"] == [0, 1, 2]


def test_train_normalization_is_immune_to_val_features(tmp_path):
    v3, measured = fixtures(tmp_path)
    data, info = prepare_dev_candidate(v3, measured, window=8)
    x = data["gps_features"].copy()
    x[168:] += 10000
    assert np.array_equal(data["train_median"],
                          np.median(x[:168], axis=0).astype(np.float32))
    p25, p75 = np.percentile(x[:168], [25, 75], axis=0)
    scale = np.asarray(p75 - p25, dtype=np.float32)
    scale[scale < 1e-6] = 1.0
    assert np.array_equal(data["train_iqr"], scale)


def test_dev_candidate_write_is_new_directory_only(tmp_path):
    v3, measured = fixtures(tmp_path)
    original = [hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (v3, measured)]
    arrays, info = prepare_dev_candidate(v3, measured, window=8)
    base = tmp_path / "new_outputs"
    dest = base / "dev_only_gps"
    data_file, meta_file = write_candidate(
        dest, arrays, info, allowed_parent=base)
    assert data_file.is_file() and meta_file.is_file()
    with np.load(data_file, allow_pickle=False) as d:
        assert set(d.files) == set(arrays)
        assert np.array_equal(d["gps_features"], arrays["gps_features"])
        assert "clean_gps_local" not in d.files
        assert "severity" not in d.files
        assert "test_anchors" not in d.files
        assert "test_targets" not in d.files
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["development_only"]
    assert not meta["approved_for_final_training"]
    with pytest.raises(FileExistsError, match="overwrite"):
        write_candidate(dest, arrays, info, allowed_parent=base)
    assert original == [
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (v3, measured)]


def test_wrong_label_source_fails_before_creation(tmp_path):
    v3, measured = fixtures(tmp_path, bad_label=True)
    with pytest.raises(ValueError, match="preflight"):
        prepare_dev_candidate(v3, measured, window=8)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "degraded.npz", "v3.npz",
    ]


def test_dest_must_be_inside_allowed_parent(tmp_path):
    v3, measured = fixtures(tmp_path)
    arrays, info = prepare_dev_candidate(v3, measured, window=8)
    with pytest.raises(ValueError, match="inside"):
        write_candidate(tmp_path / "escape", arrays, info,
                        allowed_parent=tmp_path / "restricted")
    assert not (tmp_path / "escape").exists()


def test_stage3_final_training_claim_cannot_be_written(tmp_path):
    v3, measured = fixtures(tmp_path)
    arrays, info = prepare_dev_candidate(v3, measured, window=8)
    info["approved_for_final_training"] = True
    with pytest.raises(ValueError, match="development-only"):
        write_candidate(tmp_path / "out" / "archive", arrays, info,
                        allowed_parent=tmp_path / "out")
