from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.isj_v8 import run_dev_feedforward_fgo as v8
from experiments.isj_v8.audit_forecast_dev import EXPECTED_ANCHORS


def _inputs():
    p = np.empty((680, 3), dtype=np.float64)
    p[:, 0] = np.linspace(.2, .8, 680)
    p[:, 1] = np.linspace(.35, .65, 680)
    p[:, 2] = np.linspace(.4, .6, 680)
    fit = {"sigma_fixed_xyz": np.array([2., 3., 4.])}
    return EXPECTED_ANCHORS.copy(), p, fit


def _mock_noise_mapping(monkeypatch):
    def map_model(reliability, fitted):
        r = np.asarray(reliability, dtype=np.float64)
        return np.column_stack([r, r, r]) + 2.0
    def cap_model(sigmas, fixed):
        return np.asarray(sigmas, dtype=np.float64).copy()
    monkeypatch.setattr(v8, "axis_sigma_from_prediction", map_model)
    monkeypatch.setattr(v8, "cap_axis_information", cap_model)


def test_v8_3_suffix_alignment_and_uniform_two_frame_cold_start(monkeypatch):
    _mock_noise_mapping(monkeypatch)
    anchors, p, fit = _inputs()
    d = v8.sigma_cases_from_frozen_predictions(anchors, p, fit)
    assert list(d) == list(v8.MODEL_METHODS)
    for sigmas in d.values():
        assert sigmas.shape == (680, 3)
        np.testing.assert_array_equal(
            sigmas[:2], np.tile(fit["sigma_fixed_xyz"], (2, 1)))
    assert d["h0_tminus1_persistence"][-1, 0] == pytest.approx(
        2. + p[-2, 0])
    assert d["h1_tminus1_causal_prior"][-1, 0] == pytest.approx(
        2. + p[-2, 1])
    assert d["h2_tminus2_causal_prior"][-1, 0] == pytest.approx(
        2. + p[-3, 2])
    assert d["h1h2_equal_causal_prior"][-1, 0] == pytest.approx(
        2. + .5 * (p[-2, 1] + p[-3, 2]))
    assert d["h0_current_observation_available"][-1, 0] == pytest.approx(
        2. + p[-1, 0])


def test_v8_3_preobserved_prior_excludes_current_h0(monkeypatch):
    _mock_noise_mapping(monkeypatch)
    a, p, fit = _inputs()
    d = v8.sigma_cases_from_frozen_predictions(a, p, fit)
    changed = p.copy()
    changed[-1, 0] += .05
    revised = v8.sigma_cases_from_frozen_predictions(a, changed, fit)
    for key in d:
        if key == "h0_current_observation_available":
            assert not np.array_equal(d[key], revised[key])
        else:
            np.testing.assert_array_equal(d[key], revised[key])


def test_v8_3_rejects_wrong_anchors_or_covariance_map(monkeypatch):
    _mock_noise_mapping(monkeypatch)
    a, p, fit = _inputs()
    bad = a.copy()
    bad[0] = 200
    with pytest.raises(ValueError, match="approved 0027"):
        v8.sigma_cases_from_frozen_predictions(bad, p, fit)
    with pytest.raises(ValueError, match="invalid train-only"):
        v8.sigma_cases_from_frozen_predictions(
            a, p, {"sigma_fixed_xyz": np.array([-1, 2, 3])}
        )
    bad = p.copy()
    bad[0, 0] = 1.4
    with pytest.raises(ValueError, match="frozen validation stream"):
        v8.sigma_cases_from_frozen_predictions(a, bad, fit)


def test_v8_3_preflight_metadata_all_cases_and_no_reference(monkeypatch, tmp_path):
    a, p, fit = _inputs()
    _mock_noise_mapping(monkeypatch)
    source = tmp_path / "degraded_sensor_data.npz"
    source.write_bytes(b"synthetic-sensor-data")
    monkeypatch.setattr(v8, "SOURCE", source)
    monkeypatch.setattr(v8, "DEGRADED", tmp_path)
    (tmp_path / "degraded_lidar_factor_data.npz").write_bytes(b"lidar")
    (tmp_path / "degraded_camera_factor_data.npz").write_bytes(b"camera")
    monkeypatch.setattr(v8, "read_policy", lambda: {
        "raw_sources": {
            "degraded_sensor": {"sha256": "DEGRADED_H"}
        }
    })
    monkeypatch.setattr(v8, "validate_policy", lambda p: None)
    monkeypatch.setattr(v8, "sha256", lambda p: (
        "DEGRADED_H" if Path(p) == source
        else "CHECKPOINT_H" if Path(p).name == "best.pt"
        else "PRED_H"
    ))
    monkeypatch.setattr(v8, "read_candidate", lambda: {
        "anchors": a, "archive_sha256": "DEV_H",
    })
    monkeypatch.setattr(v8, "load_train_axis_noise", lambda p: fit)
    model_root = tmp_path / "models"
    folder = model_root / "gru" / "seed_43"
    folder.mkdir(parents=True)
    (folder / "best.pt").write_bytes(b"known-synthetic-checkpoint")
    monkeypatch.setattr(v8, "CASES", {
        "gru96_r": (model_root, "gru", 114630, "baseline", "CHECKPOINT_H"),
    })
    expected_cases = ("axis_fixed",) + tuple(
        "gru96_r__" + method for method in v8.MODEL_METHODS
    )
    monkeypatch.setattr(v8, "ALL_CASES", expected_cases)
    monkeypatch.setattr(v8, "read_run", lambda *args: {
        "metrics": {
            "parameter_count": 114630, "run_family": "baseline",
            "config": {"heads": [0, 1, 2], "window": 64},
        },
        "arrays": {
            "anchors": a, "reliability": p,
            # Any synthetic oracle target must not be passed to mapping.
            "synthetic_targets": np.full((680, 3), .3),
        },
    })
    original_files = sorted(str(f) for f in tmp_path.rglob("*"))
    manifest, sigmas = v8.prepare_cases()
    assert manifest["development_only"] is True
    assert manifest["heldout_0016_0020_accessed"] is False
    assert list(sigmas) == list(expected_cases)
    assert manifest["all_cases_same_startup_fixed_frames"] == 2
    assert sorted(str(f) for f in tmp_path.rglob("*")) == original_files

    # Regression: generate producer metadata through the real prepare_cases()
    # and pass a synthetic, all-stable solve completion to the actual auditor.
    # This catches schema typos that a stand-alone mock JSON fixture misses.
    from experiments.isj_v8 import audit_dev_fgo_results as auditor
    monkeypatch.setattr(auditor, "CASES", expected_cases)
    monkeypatch.setattr(auditor, "MODELS", ("gru96_r",))
    assert manifest["all_cases_use_same_frame_range"] == [3180, 3860]
    completed = manifest.copy()
    completed["status"] = "COMPLETE_EXPLORATORY_0027_DEV_ONLY"
    completed["all_cases_stable"] = True
    completed["evaluation_uses_correlated_oxTS_reference"] = True
    completed["factors_per_graph"] = auditor.FACTORS.copy()
    completed["results"] = {
        case: {
            "solve": {
                "stabilized": True,
                "objective": 100.0,
                "passes": [{
                    "stabilized": True,
                    "position_shift_rmse_m": 0.001,
                    "relative_objective_reduction": 1e-7,
                }],
            },
            "evaluation": {metric: 2.0 for metric in auditor.METRICS},
        }
        for case in expected_cases
    }
    summary = auditor.audit_report(completed)
    assert summary["case_count"] == len(expected_cases)
    assert summary["all_stable"]
    assert sorted(str(f) for f in tmp_path.rglob("*")) == original_files


def test_v8_3_preflight_rejects_tampered_source_before_mapping(
        monkeypatch, tmp_path):
    source = tmp_path / "bad_source.npz"
    source.write_bytes(b"wrong")
    monkeypatch.setattr(v8, "SOURCE", source)
    monkeypatch.setattr(v8, "sha256", lambda p: "WRONG")
    monkeypatch.setattr(v8, "read_policy", lambda: {
        "raw_sources": {"degraded_sensor": {"sha256": "EXPECTED"}}
    })
    monkeypatch.setattr(v8, "validate_policy", lambda policy: None)
    with pytest.raises(ValueError, match="SHA mismatch"):
        v8.prepare_cases()


def test_v8_3_graph_refuses_preexisting_output_without_reads(
        monkeypatch, tmp_path):
    dest = tmp_path / "existing"
    dest.mkdir()
    monkeypatch.setattr(v8, "DEST", dest)
    monkeypatch.setattr(
        v8, "prepare_cases",
        lambda: pytest.fail("should not prepare cases if output exists"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        v8.evaluate_development_graph()


def test_v8_3_cli_requires_explicit_graph_token(monkeypatch):
    called = []
    monkeypatch.setattr(
        v8, "evaluate_development_graph",
        lambda: called.append("run")
    )
    with pytest.raises(SystemExit):
        v8.main(["--run-graph"])
    with pytest.raises(SystemExit):
        v8.main(["--run-graph", "--token", "SEALED_V7_0016_0020"])
    assert called == []
    v8.main(["--run-graph", "--token", v8.EXPLICIT_TOKEN])
    assert called == ["run"]


def test_v8_3_cli_preflight_only_uses_preparation(monkeypatch, capsys):
    monkeypatch.setattr(v8, "prepare_cases", lambda: (
        {"development_only": True}, {"axis_fixed": np.ones((680, 3))}
    ))
    monkeypatch.setattr(
        v8, "evaluate_development_graph",
        lambda: pytest.fail("preflight cannot run optimizer"),
    )
    v8.main(["--preflight"])
    assert "V8_DEV_FGO_PREFLIGHT_OK" in capsys.readouterr().out
