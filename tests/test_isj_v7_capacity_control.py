from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.isj_v7.compare_capacity_control import (
    aggregate_seeds, check_triplet, summarize_one_seed,
)
from experiments.isj_v7.model_dev_gps import (
    ReliabilityForecastV7, parameter_count,
)
from experiments.isj_v7 import train_gps_dev


def _triplet(*, seed=43):
    n = 680
    target = np.full((n, 3), 0.55, dtype=np.float32)
    anchors = np.arange(3180, 3860, dtype=np.int64)
    settings = {
        "seed": seed, "hidden": 96, "layers": 2,
        "dropout": .1, "lr": .001,
        "weight_decay": .0001, "batch_size": 64,
        "epochs": 40, "patience": 8, "min_delta": .0001,
        "input_clip": 20.0, "window": 64, "heads": [0, 1, 2],
    }
    models = {
        "gru_96": ("gru", 96, 114630, .16, "baseline"),
        "gru_106": ("gru", 106, 139290, .14, "capacity_gru_h106"),
        "mamba_96": ("mamba", 96, 139590, .12, "baseline"),
    }
    triplet = {}
    for label, (backend, hidden, params, error, family) in models.items():
        cfg = dict(settings, hidden=hidden)
        prediction = np.full((n, 3), 0.55 + error, dtype=np.float32)
        triplet[label] = {
            "metrics": {
                "backend": backend,
                "parameter_count": params,
                "run_family": family,
                "config": cfg,
                "dataset_sha256": "a" * 64,
            },
            "arrays": {
                "synthetic_targets": target.copy(),
                "anchors": anchors.copy(),
                "reliability": prediction,
            },
        }
    return triplet


def test_gru_106_is_close_to_mamba_capacity():
    gru96 = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=96, layers=2)
    gru106 = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=106, layers=2)
    assert parameter_count(gru96) == 114630
    assert parameter_count(gru106) == 139290
    assert 139590 - parameter_count(gru106) == 300
    assert abs(139590 - parameter_count(gru106)) / 139590 < .005


def test_capacity_control_labels_and_gain():
    tri = _triplet()
    check_triplet(tri, seed=43)
    row = summarize_one_seed(tri, seed=43)
    assert row["development_only"] is True
    assert row["independent_test_run"] is False
    h = row["heads"]["h0"]
    assert h["gru_106"]["MAE"] < h["gru_96"]["MAE"]
    assert h["mamba_96"]["MAE"] < h["gru_106"]["MAE"]
    assert h["mamba_improvement_over_capacity_gru_MAE_pct"] > 0


def test_capacity_guard_rejects_fake_106_provenance():
    tri = _triplet()
    tri["gru_106"]["metrics"]["run_family"] = "baseline"
    with pytest.raises(ValueError, match="not from capacity-control"):
        check_triplet(tri, 43)


def test_capacity_comparison_rejects_other_training_settings():
    tri = _triplet()
    tri["mamba_96"]["metrics"]["config"]["lr"] = .01
    with pytest.raises(ValueError, match="settings differ"):
        check_triplet(tri, 43)


def test_capacity_comparison_rejects_changed_target():
    tri = _triplet()
    tri["gru_106"]["arrays"]["synthetic_targets"][0, 0] += .001
    with pytest.raises(ValueError, match="validation targets"):
        check_triplet(tri, 43)


def test_aggregate_seeds_disallows_duplicate_runs():
    first = summarize_one_seed(_triplet(seed=43), 43)
    second = summarize_one_seed(_triplet(seed=44), 44)
    summary = aggregate_seeds([first, second])
    assert summary["schema"] == "isj_v7_dev_capacity_control_v1"
    assert summary["independent_drives"] is False
    assert summary["seeds"] == [43, 44]
    assert summary["model_parameters"]["gru_106"] == 139290
    assert summary["horizons"]["h0"][
        "mamba_gain_over_capacity_gru_MAE_pct"] > 0
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_seeds([first, first])


def _args(**kwargs):
    raw = {
        "backend": "gru", "device": "cpu", "seed": 43,
        "hidden": 106, "layers": 2, "dropout": .1,
        "lr": .001, "weight_decay": .0001,
        "batch_size": 64, "epochs": 2, "patience": 2,
        "min_delta": .0001, "input_clip": 20.,
        "smoke_only": True, "capacity_control": True,
        "output": train_gps_dev.DEFAULT_OUTPUT,
    }
    raw.update(kwargs)
    return SimpleNamespace(**raw)


def test_capacity_mode_rejects_wrong_model_before_loading_data(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("data loader must not be called")
    monkeypatch.setattr(train_gps_dev, "load_dev_data", forbidden)
    with pytest.raises(ValueError, match="requires GRU hidden=106"):
        train_gps_dev.run(_args(backend="mamba"))
    with pytest.raises(ValueError, match="requires GRU hidden=106"):
        train_gps_dev.run(_args(hidden=96))
    with pytest.raises(ValueError, match="requires GRU hidden=106"):
        train_gps_dev.run(_args(layers=1))
