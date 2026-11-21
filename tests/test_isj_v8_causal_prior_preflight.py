from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_v8 import causal_prior_dev as prior
from experiments.isj_v8.audit_forecast_dev import EXPECTED_ANCHORS


def _fixture():
    labels = np.full(682, .35, dtype=np.float32)
    labels[100:132] = .80
    labels[430:463] = .10
    y = labels[np.arange(680)[:, None] + np.arange(3)[None, :]]
    p = (.99 * y + .005).astype(np.float32)
    u = np.full_like(p, .6)
    return EXPECTED_ANCHORS.copy(), y, p, u


def test_v8_causal_prior_exact_time_index_and_alignment():
    a, y, p, u = _fixture()
    ids, src = prior.preobservation_candidates(a, p)
    assert np.array_equal(ids, a[2:])
    assert len(ids) == 678
    np.testing.assert_array_equal(
        src["forecast_h1_t_minus_1"], p[1:-1, 1])
    np.testing.assert_array_equal(
        src["forecast_h2_t_minus_2"], p[:-2, 2])
    np.testing.assert_array_equal(
        src["persistence_h0_t_minus_1"], p[1:-1, 0])
    np.testing.assert_array_equal(
        src["persistence_h0_t_minus_2"], p[:-2, 0])
    np.testing.assert_array_equal(
        src["post_observation_h0_t_diagnostic_only"], p[2:, 0])
    np.testing.assert_allclose(
        src["forecast_h1h2_equal_mean"],
        .5 * (p[1:-1, 1] + p[:-2, 2]), rtol=1e-6)


def test_v8_causal_priors_do_not_use_current_t_h0():
    a, y, p, u = _fixture()
    _, src = prior.preobservation_candidates(a, p)
    perturbed = p.copy()
    perturbed[-1, 0] = .98
    _, revised = prior.preobservation_candidates(a, perturbed)
    for name in src:
        if name == "post_observation_h0_t_diagnostic_only":
            assert not np.array_equal(src[name], revised[name])
        else:
            np.testing.assert_array_equal(src[name], revised[name])


def test_v8_causal_prior_score_only_uses_development_labels():
    a, y, p, u = _fixture()
    baseline = prior.score_causal_priors(a, y, p, u)
    assert baseline["evaluated_anchor_count"] == 678
    assert baseline["evaluated_anchor_start_inclusive"] == 3182
    assert baseline["evaluated_anchor_end_exclusive"] == 3860
    assert baseline["prior_only_uses_observations_through"] == "t_minus_1"
    assert baseline["two_head_blend_weights"] == [.5, .5]
    assert len(baseline["performance"]) == 6
    assert baseline["performance"]["forecast_h1_t_minus_1"]["rmse"] >= 0
    assert 0 < baseline["retrospective_target_selected_transitions"]["count"] < 678
    assert baseline["retrospective_target_selected_transitions"]["not_online_or_confirmatory"]


def test_v8_causal_prior_scoring_no_mutations():
    a, y, p, u = _fixture()
    originals = [arr.copy() for arr in (a, y, p, u)]
    prior.score_causal_priors(a, y, p, u)
    for actual, original in zip((a, y, p, u), originals):
        np.testing.assert_array_equal(actual, original)


def test_v8_causal_prior_rejects_nonconsecutive_val_and_bad_predictions():
    a, y, p, u = _fixture()
    bad = a.copy()
    bad[80] += 3
    with pytest.raises(ValueError, match="validation stream"):
        prior.preobservation_candidates(bad, p)
    corrupted = p.copy()
    corrupted[80, 2] = np.nan
    with pytest.raises(ValueError, match="validation stream"):
        prior.preobservation_candidates(a, corrupted)
    with pytest.raises(ValueError, match="validation stream"):
        prior.preobservation_candidates(a, p[:-1])


def test_v8_causal_prior_rejects_bad_evidence_and_bad_target_horizon():
    a, y, p, u = _fixture()
    bad = u.copy()
    bad[3, 1] = 0
    with pytest.raises(ValueError, match="evidential proxy"):
        prior.score_causal_priors(a, y, p, bad)
    mislabeled = y.copy()
    mislabeled[100, 1] = .01
    with pytest.raises(ValueError, match="target shift"):
        prior.score_causal_priors(a, mislabeled, p, u)


def test_v8_causal_prior_readonly_frozen_sources_and_no_files_written(
        monkeypatch, tmp_path):
    a, y, p, u = _fixture()
    candidate = {
        "anchors": a, "targets": y, "archive_sha256": "DEV",
    }
    root = tmp_path / "dev"
    folder = root / "gru" / "seed_43"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"fake-ckpt")
    monkeypatch.setattr(prior, "CASES", {
        "gru96_r": (root, "gru", 114630, "baseline", "SHA"),
    })
    monkeypatch.setattr(prior, "read_candidate", lambda: candidate)
    monkeypatch.setattr(prior, "read_run", lambda *args: {
        "metrics": {
            "parameter_count": 114630,
            "run_family": "baseline",
            "config": {"heads": [0, 1, 2], "window": 64},
        },
        "arrays": {
            "anchors": a, "synthetic_targets": y,
            "reliability": p, "evidence_uncertainty": u,
        },
    })
    monkeypatch.setattr(prior, "sha256", lambda path: (
        "SHA" if path.name == "best.pt" else "PRED_SHA"
    ))
    files_before = [
        (str(path), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*") if path.is_file()
    ]
    report = prior.run_read_only()
    assert list(report["models"]) == ["gru96_r"]
    assert report["no_training"] and report["no_graph"]
    assert report["no_output_written"]
    assert report["heldout_content_read"] is False
    assert [
        (str(path), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*") if path.is_file()
    ] == files_before


def test_v8_causal_prior_refuses_frozen_checkpoint_hash_mismatch(
        monkeypatch, tmp_path):
    a, y, p, u = _fixture()
    root = tmp_path / "dev"
    folder = root / "gru" / "seed_43"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"tampered")
    monkeypatch.setattr(prior, "CASES", {
        "gru96_r": (root, "gru", 114630, "baseline", "EXPECTED"),
    })
    monkeypatch.setattr(prior, "read_candidate", lambda: {
        "anchors": a, "targets": y, "archive_sha256": "DEV",
    })
    monkeypatch.setattr(prior, "read_run", lambda *args: {
        "metrics": {
            "parameter_count": 114630,
            "run_family": "baseline",
            "config": {"heads": [0, 1, 2], "window": 64},
        },
        "arrays": {
            "anchors": a, "synthetic_targets": y,
            "reliability": p, "evidence_uncertainty": u,
        },
    })
    monkeypatch.setattr(prior, "sha256", lambda path: "WRONG")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prior.run_read_only()


def test_v8_causal_prior_cli_only_preflight(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        prior.main([])
    with pytest.raises(SystemExit):
        prior.main(["--run-graph"])
    monkeypatch.setattr(
        prior, "run_read_only",
        lambda: {"status": "synthetic_dev_only"})
    prior.main(["--preflight"])
    assert "V8_CAUSAL_PRIOR_DEV_PREFLIGHT_OK" in capsys.readouterr().out
