from __future__ import annotations

import hashlib
import numpy as np
import pytest

from experiments.isj_v7.protocol import (
    aligned_targets, audit_archive, plan_splits, robust_train_scaler,
)


def test_0027_purged_boundaries_have_no_future_label_crossing():
    plan = plan_splits(4544, window=64, horizon=3)
    assert (plan.train_end, plan.val_end) == (3180, 3862)
    expected = {
        "train": (63, 3177, 3115, 3179),
        "val": (3180, 3859, 680, 3861),
        "test": (3862, 4541, 680, 4543),
    }
    for part, (first, last, count, last_target) in expected.items():
        t = plan.indices(part)
        lo, hi = plan.boundary(part)
        assert (t[0], t[-1], len(t), t[-1] + 2) == (
            first, last, count, last_target)
        assert t.min() >= lo if part != "train" else t.min() >= 63
        assert np.all(t + 2 < hi)
    assert plan.offsets == (0, 1, 2)


def test_targets_are_t_tplus1_tplus2_and_never_padded_future():
    cur = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    got = aligned_targets(cur, np.array([7, 8, 47]), horizon=3)
    assert np.array_equal(got, np.array([
        cur[7:10], cur[8:11], cur[47:50]]))
    with pytest.raises(ValueError, match="exceed"):
        aligned_targets(cur, np.array([48]), horizon=3)
    with pytest.raises(ValueError, match="invalid"):
        aligned_targets(np.array([0.2, np.nan]), np.array([0]))


def test_normalizer_only_fits_train_prefix():
    x = np.arange(600, dtype=np.float32).reshape(200, 3)
    a = robust_train_scaler(x, 140)
    x[140:] += 1e8
    b = robust_train_scaler(x, 140)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.all(a[1] > 0)
    with pytest.raises(ValueError):
        robust_train_scaler(x, 0)


def test_empty_and_invalid_configuration_rejected():
    for kw in ({"window": 128}, {"horizon": 80},
               {"train_fraction": 0.9, "val_fraction": 0.8}):
        with pytest.raises(ValueError):
            plan_splits(100, **kw)
    with pytest.raises(ValueError, match="unknown"):
        plan_splits(4544).indices("oops")


def test_readonly_preflight_manifest_blocks_legacy_labels(tmp_path):
    n = 120
    arrays = {}
    for sensor, dim in (("gps", 21), ("imu", 20), ("camera", 16)):
        arrays[f"{sensor}_features"] = np.tile(
            np.arange(dim, dtype=np.float32)[None, :], (n, 1))
        arrays[f"{sensor}_current_factor_reliability"] = np.linspace(
            0.1, 0.9, n, dtype=np.float32)
        arrays[f"{sensor}_future_factor_reliability"] = np.ones(
            n, dtype=np.float32)  # deliberately wrong; must be ignored
    p = tmp_path / "v3.npz"
    np.savez_compressed(p, **arrays)
    digest_before = hashlib.sha256(p.read_bytes()).hexdigest()
    report = audit_archive(p, window=8, horizon=3)
    assert digest_before == hashlib.sha256(p.read_bytes()).hexdigest()
    assert report["source_sha256"] == digest_before
    assert report["legacy_future_column_ignored"] is True
    assert report["approved_for_final_training"] is False
    assert report["head_offsets"] == [0, 1, 2]
    assert report["sensors"]["gps"]["samples"]["train"]["count"] > 0
    assert report["sensors"]["camera"]["samples"]["val"]["last_target"] < report["val_end_exclusive"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v3.npz"]


def test_missing_or_corrupt_source_fails_without_substitution(tmp_path):
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        audit_archive(missing)
    assert not missing.exists()
    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, gps_features=np.zeros((120, 21), dtype=np.float32))
    with pytest.raises(KeyError):
        audit_archive(wrong, window=8)
