from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_v8 import audit_forecast_dev as v8


def _fixture(with_changes=True):
    labels = np.full(682, 0.3, dtype=np.float32)
    if with_changes:
        labels[130:180] = 0.8
        labels[390:430] = 0.1
    anchors = v8.EXPECTED_ANCHORS.copy()
    targets = labels[np.arange(680)[:, None] + np.arange(3)[None, :]]
    # A valid hypothetical predictor exactly tracks all future labels.
    pred = (0.99 * targets + 0.005).astype(np.float32)
    uncert = np.full((680, 3), 0.6, dtype=np.float32)
    return anchors, targets, pred, uncert


def test_v8_exact_temporal_target_alignment_and_offsets():
    anchors, y, pred, uncert = _fixture()
    v8.validate_development_alignment(anchors, y)
    assert np.array_equal(y[:-1, 1], y[1:, 0])
    assert np.array_equal(y[:-2, 2], y[2:, 0])
    scores = v8.score_development_forecasts(anchors, y, pred, uncert)
    assert scores["h1"]["target_frame_offset"] == 1
    assert scores["h2"]["target_frame_offset"] == 2
    assert scores["h1"]["forecast_beats_persistence_rmse"]
    assert scores["h2"]["forecast_beats_persistence_rmse"]


def test_v8_development_transition_subset_is_diagnostic_only():
    anchors, y, pred, u = _fixture(with_changes=True)
    scores = v8.score_development_forecasts(anchors, y, pred, u)
    for h in (1, 2):
        r = scores[f"h{h}"]["oracle_defined_change_subset"]
        assert 0 < r["count"] < 680
        assert r["forecast"]["rmse"] < r["causal_h0_persistence"]["rmse"]
        assert r["predicted_change_sign_accuracy"] == 1.0
        assert "development_only" in r["use"]


def test_v8_zero_change_subset_is_explicit_null_not_a_spurious_score():
    anchors, y, pred, u = _fixture(with_changes=False)
    scores = v8.score_development_forecasts(anchors, y, pred, u)
    for h in (1, 2):
        r = scores[f"h{h}"]["oracle_defined_change_subset"]
        assert r["count"] == 0
        assert r["forecast"] is None
        assert r["causal_h0_persistence"] is None
        assert r["predicted_change_sign_accuracy"] is None


def test_v8_rejects_target_horizon_misalignment_and_forbidden_anchors():
    anchors, y, pred, u = _fixture()
    corrupted = y.copy()
    corrupted[4, 1] = 0.2
    with pytest.raises(ValueError, match="target shift"):
        v8.score_development_forecasts(anchors, corrupted, pred, u)
    wrong = anchors.copy()
    wrong[0] = 3862
    with pytest.raises(ValueError, match="anchors"):
        v8.score_development_forecasts(wrong, y, pred, u)
    with pytest.raises(ValueError, match="anchors"):
        v8.score_development_forecasts(anchors[:-1], y, pred, u)


def test_v8_rejects_invalid_prediction_and_evidence_domains():
    anchors, y, pred, u = _fixture()
    bad = pred.copy()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="reliability"):
        v8.score_development_forecasts(anchors, y, bad, u)
    bad = pred.copy()
    bad[0, 2] = -0.1
    with pytest.raises(ValueError, match="reliability"):
        v8.score_development_forecasts(anchors, y, bad, u)
    bad_u = u.copy()
    bad_u[0, 0] = 0.
    with pytest.raises(ValueError, match="evidence"):
        v8.score_development_forecasts(anchors, y, pred, bad_u)


def test_v8_forecast_scoring_does_not_mutate_arrays():
    anchors, y, p, u = _fixture()
    before = [x.copy() for x in (anchors, y, p, u)]
    v8.score_development_forecasts(anchors, y, p, u)
    for actual, previous in zip((anchors, y, p, u), before):
        assert np.array_equal(actual, previous)


def test_v8_readonly_gate_accepts_only_frozen_case_and_metadata(
        monkeypatch, tmp_path):
    anchors, y, p, u = _fixture()
    candidate = {
        "anchors": anchors, "targets": y, "archive_sha256": "DEVDATA",
    }
    monkeypatch.setattr(v8, "read_candidate", lambda: candidate)
    root_a = tmp_path / "baseline"
    root_b = tmp_path / "capacity"
    frozen = {
        "gru96_r": (root_a, "gru", 114630, "baseline", "CKPT_A"),
        "gru106_r": (root_b, "gru", 139290, "capacity_gru_h106", "CKPT_B"),
        "mamba_r": (root_a, "mamba", 139590, "baseline", "CKPT_C"),
    }
    monkeypatch.setattr(v8, "CASES", frozen)
    for name, (root, backend, _, _, _) in frozen.items():
        folder = root / backend / "seed_43"
        folder.mkdir(parents=True)
        (folder / "best.pt").write_bytes(b"synthetic checkpoint")
    def fake_read(root, backend, seed, c):
        row = next(row for row in frozen.values()
                   if row[0] == root and row[1] == backend)
        return {
            "metrics": {
                "parameter_count": row[2], "run_family": row[3],
                "config": {"window": 64, "heads": [0, 1, 2]},
            },
            "arrays": {
                "anchors": anchors, "synthetic_targets": y,
                "reliability": p, "evidence_uncertainty": u,
            },
        }
    monkeypatch.setattr(v8, "read_run", fake_read)
    monkeypatch.setattr(
        v8, "sha256",
        lambda f: (next((row[4] for row in frozen.values()
                         if str(f).startswith(str(row[0] / row[1]))
                         and f.name == "best.pt"), "VAL_PRED_HASH")),
    )
    prefiles = sorted(str(p) for p in tmp_path.rglob("*"))
    report = v8.run_read_only()
    assert sorted(report["models"]) == sorted(frozen)
    assert report["heldout_sensor_content_read"] is False
    assert report["model_retrained"] is False
    assert report["factor_graph_evaluated"] is False
    assert report["forecast_heads_not_yet_deployed"] == [1, 2]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == prefiles


def test_v8_gate_rejects_unknown_checkpoint_hash_before_any_scoring(
        monkeypatch, tmp_path):
    anchors, y, p, u = _fixture()
    root = tmp_path / "dev"
    folder = root / "gru" / "seed_43"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"wrong")
    monkeypatch.setattr(v8, "CASES", {
        "gru96_r": (root, "gru", 114630, "baseline", "KNOWN"),
    })
    monkeypatch.setattr(v8, "read_candidate", lambda: {
        "anchors": anchors, "targets": y, "archive_sha256": "DEV",
    })
    monkeypatch.setattr(v8, "read_run", lambda *a: {
        "metrics": {
            "parameter_count": 114630, "run_family": "baseline",
            "config": {"window": 64, "heads": [0, 1, 2]},
        },
        "arrays": {
            "anchors": anchors, "synthetic_targets": y,
            "reliability": p, "evidence_uncertainty": u,
        },
    })
    monkeypatch.setattr(v8, "sha256", lambda f: "NOT_KNOWN")
    with pytest.raises(ValueError, match="SHA256"):
        v8.run_read_only()


def test_v8_cli_has_no_persisting_or_training_flags(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        v8.main([])
    with pytest.raises(SystemExit):
        v8.main(["--write-report"])
    called = []
    monkeypatch.setattr(
        v8, "run_read_only",
        lambda: called.append(True) or {"development_only": True},
    )
    v8.main(["--preflight"])
    assert called == [True]
    assert "V8_DEV_HORIZON_PREFLIGHT_OK" in capsys.readouterr().out
