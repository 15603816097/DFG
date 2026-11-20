from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.isj_v7.audit_gps_canonical_dryrun import _array_sha
from experiments.isj_v7.model_dev_gps import (
    ReliabilityForecastV7, evidential_loss, parameter_count,
)
from experiments.isj_v7 import train_gps_dev as trainer
from experiments.isj_v7.supervision_gate import read_policy
from experiments.isj_v7.protocol import robust_train_scaler


def _synth_dev(tmp_path: Path):
    rng = np.random.default_rng(81)
    x = rng.normal(0, .1, size=(3862, 21)).astype(np.float32)
    a = np.arange(63, 3178, dtype=np.int64)
    b = np.arange(3180, 3860, dtype=np.int64)
    y = (np.sin(np.arange(3862) * .003) * .2 + .6).astype(np.float32)
    train_targets = np.stack([y[a], y[a + 1], y[a + 2]], axis=1)
    val_targets = np.stack([y[b], y[b + 1], y[b + 2]], axis=1)
    mean, scale = robust_train_scaler(x, 3180)
    data = dict(
        gps_features=x, train_anchors=a, train_targets=train_targets,
        val_anchors=b, val_targets=val_targets,
        train_median=mean, train_iqr=scale,
    )
    policy = read_policy()
    dest = tmp_path / "candidate"
    dest.mkdir()
    np.savez_compressed(dest / "gps_train_val.npz", **data)
    meta = {
        "schema": "isj_v7_dev_gps_same_source_v1",
        "development_only": True,
        "approved_for_final_training": False,
        "approved_for_independent_test": False,
        "do_not_treat_as_independent_truth": True,
        "source_drive": "2011_10_03_drive_0027_sync",
        "source_frames": 4544,
        "exported_feature_frames": 3862,
        "omitted_internal_test_range": [3862, 4544],
        "window": 64,
        "forecast_heads": [0, 1, 2],
        "train_end_exclusive": 3180,
        "val_end_exclusive": 3862,
        "source_archive_sha256": {
            key: policy["raw_sources"][key]["sha256"]
            for key in ("v3", "degraded_sensor")
        },
        "array_sha256_float32": {
            k: _array_sha(a) for k, a in data.items()
        },
        "array_dtypes": {k: str(a.dtype) for k, a in data.items()},
    }
    (dest / "provenance.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return dest, data, meta


def test_gru_evidential_heads_forward_backward_cpu():
    model = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=16,
        layers=1, horizon=3, dropout=0.,
    )
    x = torch.randn(4, 64, 21)
    y = torch.rand(4, 3) * .8 + .1
    out = model(x)
    for key in ("alpha", "beta", "reliability", "uncertainty"):
        assert out[key].shape == (4, 3)
        assert torch.isfinite(out[key]).all()
    assert (out["alpha"] > 1).all()
    assert (out["beta"] > 1).all()
    assert ((out["reliability"] > 0) &
            (out["reliability"] < 1)).all()
    assert ((out["uncertainty"] > 0) &
            (out["uncertainty"] < 1)).all()
    loss = evidential_loss(out, y)
    loss.backward()
    assert parameter_count(model) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_mamba_missing_dependency_fails_closed(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mamba_ssm", None)
    with pytest.raises(RuntimeError, match="NOT fall back to GRU"):
        ReliabilityForecastV7(
            backend="mamba", input_dim=21, hidden=16, layers=1,
        )


def test_loader_verifies_artifact_and_anchors(tmp_path, monkeypatch):
    dest, data, meta = _synth_dev(tmp_path)
    monkeypatch.setattr(trainer, "DEFAULT_DEST", dest)
    loaded, manifest = trainer.load_dev_data(dest)
    assert loaded["gps_features"].shape == (3862, 21)
    assert loaded["train_targets"].shape == (3115, 3)
    assert loaded["val_targets"].shape == (680, 3)
    assert manifest["archive_sha256"] == trainer.file_sha(
        dest / "gps_train_val.npz")
    ds = trainer.WindowDataset(
        loaded["gps_features"], loaded["val_anchors"],
        loaded["val_targets"], loaded["train_median"],
        loaded["train_iqr"], window=64, clip=20.,
    )
    x, y, t = ds[0]
    assert t == 3180
    assert x.shape == (64, 21) and y.shape == (3,)
    normalized = ((loaded["gps_features"][3117:3181] -
                   loaded["train_median"]) / loaded["train_iqr"])
    assert np.allclose(x.numpy(), np.clip(normalized, -20., 20.))


def test_loader_refuses_hidden_test_or_oracle_fields(tmp_path, monkeypatch):
    dest, data, meta = _synth_dev(tmp_path)
    monkeypatch.setattr(trainer, "DEFAULT_DEST", dest)
    contaminated = dict(data, clean_gps_local=np.zeros((3862, 3)))
    np.savez_compressed(dest / "gps_train_val.npz", **contaminated)
    with pytest.raises(ValueError, match="possible oracle/test leakage"):
        trainer.load_dev_data(dest)


def test_loader_refuses_tampered_features_or_targets(tmp_path, monkeypatch):
    dest, data, meta = _synth_dev(tmp_path)
    monkeypatch.setattr(trainer, "DEFAULT_DEST", dest)
    tampered = {k: v.copy() for k, v in data.items()}
    tampered["gps_features"][100, 0] += 11.0
    np.savez_compressed(dest / "gps_train_val.npz", **tampered)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        trainer.load_dev_data(dest)


def test_loader_refuses_wrong_train_val_boundary(tmp_path, monkeypatch):
    dest, data, meta = _synth_dev(tmp_path)
    monkeypatch.setattr(trainer, "DEFAULT_DEST", dest)
    meta["omitted_internal_test_range"] = [3861, 4544]
    (dest / "provenance.json").write_text(
        json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        trainer.load_dev_data(dest)


def test_dev_cpu_smoke_has_no_optimizer_output(tmp_path, monkeypatch):
    dest, data, meta = _synth_dev(tmp_path)
    monkeypatch.setattr(trainer, "DEFAULT_DEST", dest)
    out = tmp_path / "dev_outputs"
    monkeypatch.setattr(trainer, "DEFAULT_OUTPUT", out)
    args = SimpleNamespace(
        backend="gru", device="cpu", seed=42,
        hidden=16, layers=1, dropout=0.,
        lr=.001, weight_decay=.0001,
        batch_size=8, epochs=2, patience=2,
        min_delta=.0001, input_clip=20.0,
        output=out, smoke_only=True,
    )
    result = trainer.run(args)
    assert result["backend"] == "gru"
    assert result["development_only"] is True
    assert not out.exists()


def test_metrics_scoring_is_val_only_and_headwise():
    y = np.asarray([[.1, .2, .3], [.5, .6, .7]], np.float32)
    p = np.asarray([[.2, .1, .4], [.4, .7, .6]], np.float32)
    u = np.full_like(y, .4)
    train = np.ones((8, 3), dtype=np.float32) * .5
    score = trainer.score_heads(p, y, u, train)
    assert list(score) == ["h0", "h1", "h2"]
    assert score["h0"]["MAE"] == pytest.approx(.1)
    assert score["h1"]["uncertainty_mean"] == pytest.approx(.4)
    with pytest.raises(ValueError):
        trainer.score_heads(p, y, u[:, :1], train)


def test_mamba_backend_cannot_silently_use_cpu():
    with pytest.raises(RuntimeError, match="requires NVIDIA CUDA"):
        trainer.choose_device("cpu", "mamba")
    assert trainer.choose_device("cpu", "gru").type == "cpu"


def test_frozen_inference_matches_training_window_and_excludes_cold_start():
    model = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=8,
        layers=1, horizon=3, dropout=0.,
    )
    model.eval()
    features = np.arange(80 * 21, dtype=np.float32).reshape(80, 21) / 500.
    median = np.zeros(21, dtype=np.float32)
    iqr = np.ones(21, dtype=np.float32)
    anchors, r, u = trainer.predict_frozen_gps_features_causally(
        model, features, median, iqr, batch_size=7,
    )
    assert np.array_equal(anchors, np.arange(63, 80))
    assert r.shape == (17, 3)
    assert u.shape == (17, 3)
    assert np.all((r > 0) & (r < 1))
    with torch.inference_mode():
        direct = model(torch.from_numpy(features[:64][None]))
    assert np.allclose(r[0], direct["reliability"].numpy(), atol=1e-7)


def test_frozen_inference_prefix_invariance_no_future_leakage():
    torch.manual_seed(902)
    model = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=8, layers=1,
        horizon=3, dropout=.0,
    )
    rng = np.random.default_rng(23)
    full = rng.normal(size=(116, 21)).astype(np.float32)
    mean = rng.normal(size=21).astype(np.float32)
    iqr = np.ones(21, dtype=np.float32)
    aa, ar, au = trainer.predict_frozen_gps_features_causally(
        model, full, mean, iqr, batch_size=16,
    )
    tampered = full.copy()
    tampered[95:] = 1e6
    bb, br, bu = trainer.predict_frozen_gps_features_causally(
        model, tampered, mean, iqr, batch_size=3,
    )
    # t<=94 cannot use future features >=95, even across batches.
    assert np.array_equal(aa, bb)
    assert np.allclose(ar[aa <= 94], br[bb <= 94], atol=1e-6)
    assert np.allclose(au[aa <= 94], bu[bb <= 94], atol=1e-6)
    assert not np.allclose(ar[aa >= 95], br[bb >= 95])


def test_frozen_inference_invalid_dimensions_scaler_and_short_input():
    model = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=8, layers=1, dropout=0.,
    )
    x = np.ones((70, 21), dtype=np.float32)
    m = np.zeros(21, dtype=np.float32)
    q = np.ones(21, dtype=np.float32)
    invalid = [
        (x[:63], m, q, {}),
        (x[:, :20], m, q, {}),
        (x, m[:20], q, {}),
        (x, m, np.zeros(21), {}),
        (np.full_like(x, np.nan), m, q, {}),
        (x, m, q, {"window": 63}),
        (x, m, q, {"batch_size": 0}),
    ]
    for xx, mm, qq, opts in invalid:
        with pytest.raises(ValueError, match="invalid V7 causal"):
            trainer.predict_frozen_gps_features_causally(
                model, xx, mm, qq, **opts)
