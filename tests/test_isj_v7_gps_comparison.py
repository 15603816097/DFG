from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.isj_v7.compare_dev_gps import (
    block_bootstrap_delta,
    compare_seed,
    read_run,
    uncertainty_diagnostics,
)


def _write_fixture(tmp_path: Path):
    rng = np.random.default_rng(20261120)
    anchors = np.arange(3180, 3860, dtype=np.int64)
    n = len(anchors)
    synthetic_target = (
        0.5 + .2 * np.sin(np.arange(n) / 30)
    ).astype(np.float32)
    targets = np.stack(
        [synthetic_target, synthetic_target,
         synthetic_target], axis=1,
    )
    candidate = {
        "anchors": anchors,
        "targets": targets,
        "archive_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    cfg = {
        "seed": 42, "hidden": 96, "layers": 2, "dropout": .1,
        "lr": .001, "weight_decay": .0001, "batch_size": 64,
        "epochs": 40, "patience": 8, "min_delta": .0001,
        "input_clip": 20.0, "window": 64, "heads": [0, 1, 2],
    }
    for backend, noise, params in (
        ("gru", .14, 114630), ("mamba", .07, 139590)
    ):
        noise_vec = rng.normal(0, noise, size=targets.shape)
        prediction = np.clip(
            targets + noise_vec, .01, .99).astype(np.float32)
        uncertainty = np.clip(
            0.25 + 0.10 * np.abs(noise_vec), .001, .99
        ).astype(np.float32)
        val_heads = {}
        for j in range(3):
            err = prediction[:, j] - targets[:, j]
            val_heads[f"h{j}"] = {
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
            }
        output = tmp_path / backend / "seed_42"
        output.mkdir(parents=True)
        (output / "metrics.json").write_text(
            json.dumps({
                "schema": "isj_v7_dev_gps_trainval_baseline_v1",
                "backend": backend,
                "development_only": True,
                "approved_for_final_training": False,
                "independent_test_run": False,
                "validation_is_from_0027_development": True,
                "config": cfg.copy(),
                "dataset_sha256": candidate["archive_sha256"],
                "dataset_manifest_sha256":
                    candidate["manifest_sha256"],
                "val_heads": val_heads,
                "parameter_count": params,
                "device": "cuda" if backend == "mamba" else "cpu",
                "best_epoch": 1 if backend == "gru" else 8,
            }), encoding="utf-8",
        )
        np.savez_compressed(
            output / "val_predictions.npz",
            anchors=anchors, synthetic_targets=targets,
            reliability=prediction,
            evidence_uncertainty=uncertainty,
        )
    return tmp_path, candidate


def test_pairwise_comparison_validates_data_and_scores(tmp_path):
    root, candidate = _write_fixture(tmp_path)
    result = compare_seed(root, candidate, 42, block=32, repeats=100)
    assert result["schema"] == "isj_v7_gps_dev_pairwise_v1"
    assert result["development_only"] is True
    assert result["independent_drive_claim"] is False
    assert result["training_device"] == {
        "gru": "cpu", "mamba": "cuda",
    }
    assert result["parameters_are_matched"] is False
    assert result["validation_frame_count"] == 680
    for h in ("h0", "h1", "h2"):
        assert result["comparison"][h]["mamba_mae"] < (
            result["comparison"][h]["gru_mae"])
        assert result["comparison"][h][
            "mae_relative_improvement_pct"] > 0
        assert len(result["comparison"][h][
            "paired_temporal_bootstrap"][
            "exploratory_block_bootstrap_95pct_interval"]) == 2


def test_block_bootstrap_paired_sign_and_determinism():
    g = np.full(200, 0.20)
    m = np.full(200, 0.08)
    one = block_bootstrap_delta(
        g, m, block=16, repeats=32, seed=43)
    two = block_bootstrap_delta(
        g, m, block=16, repeats=32, seed=43)
    assert one == two
    assert one["mae_delta_mamba_minus_gru"] == pytest.approx(-0.12)
    assert one["exploratory_block_bootstrap_95pct_interval"] == (
        pytest.approx([-0.12, -0.12])
    )


def test_uncertainty_coverage_and_domain():
    rng = np.random.default_rng(17)
    p = rng.uniform(0.3, 0.7, size=(128, 3))
    y = np.clip(p + rng.normal(0, .1, size=p.shape), 0, 1)
    u = rng.uniform(.15, .50, size=p.shape)
    results = uncertainty_diagnostics(p, u, y)
    assert set(results) == {"h0", "h1", "h2"}
    for v in results.values():
        assert 0 <= v["assumed_beta_80pct_interval_coverage"] <= 1
        assert v["assumed_beta_80pct_interval_mean_width"] > 0
    with pytest.raises(ValueError):
        uncertainty_diagnostics(p, -u, y)


def test_modified_validation_anchors_are_rejected(tmp_path):
    root, candidate = _write_fixture(tmp_path)
    p = root / "mamba" / "seed_42" / "val_predictions.npz"
    with np.load(p, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["anchors"][0] = 3181
    np.savez_compressed(p, **arrays)
    with pytest.raises(ValueError, match="validation anchors"):
        compare_seed(root, candidate, 42, repeats=10)


def test_modified_validation_targets_are_rejected(tmp_path):
    root, candidate = _write_fixture(tmp_path)
    p = root / "gru" / "seed_42" / "val_predictions.npz"
    with np.load(p, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["synthetic_targets"][0, 0] += .1
    np.savez_compressed(p, **arrays)
    with pytest.raises(ValueError, match="validation labels"):
        read_run(root, "gru", 42, candidate)


def test_metrics_tampering_cannot_be_labeled_a_result(tmp_path):
    root, candidate = _write_fixture(tmp_path)
    file = root / "gru" / "seed_42" / "metrics.json"
    metrics = json.loads(file.read_text(encoding="utf-8"))
    metrics["val_heads"]["h0"]["MAE"] = .00001
    file.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="saved metrics"):
        compare_seed(root, candidate, 42, repeats=10)


def test_mismatched_training_hyperparams_fail_closed(tmp_path):
    root, candidate = _write_fixture(tmp_path)
    file = root / "mamba" / "seed_42" / "metrics.json"
    metrics = json.loads(file.read_text(encoding="utf-8"))
    metrics["config"]["hidden"] = 112
    file.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="different run configurations"):
        compare_seed(root, candidate, 42, repeats=10)


def test_missing_backend_run_is_not_inferred(tmp_path):
    root, candidate = _write_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        compare_seed(root, candidate, 43, repeats=10)


def test_wrong_source_digest_is_rejected(tmp_path):
    root, candidate = _write_fixture(tmp_path)
    candidate["archive_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="training provenance"):
        compare_seed(root, candidate, 42, repeats=10)
