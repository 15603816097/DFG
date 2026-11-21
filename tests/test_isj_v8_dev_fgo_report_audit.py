from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.isj_v8 import audit_dev_fgo_results as a


def _fixture():
    data = {
        "schema": "v8_0027_dev_feedforward_fgo_v1",
        "status": "COMPLETE_EXPLORATORY_0027_DEV_ONLY",
        "development_only": True,
        "independent_position_ground_truth": False,
        "heldout_0016_0020_accessed": False,
        "forecast_head_future_observation_leakage": False,
        "predictive_feedback_used": False,
        "batch_Levenberg_Marquardt_not_iSAM2": True,
        "all_cases_same_frame_range": [3180, 3860],
        "all_cases_same_startup_fixed_frames": 2,
        "checkpoint_seed": 43,
        "all_cases_stable": True,
        "evaluation_uses_correlated_oxTS_reference": True,
        "comparisons_frozen_before_dev_FGO": list(a.CASES),
        "factors_per_graph": a.FACTORS.copy(),
        "results": {}, "case_sigmas": {},
    }
    for case in a.CASES:
        if case == "axis_fixed":
            rmse = 5.0
        elif case.endswith("h0_tminus1_persistence"):
            rmse = 2.0
        elif case.endswith("h1_tminus1_causal_prior"):
            rmse = 1.8
        elif case.endswith("h1h2_equal_causal_prior"):
            rmse = 2.2
        elif case.endswith("h2_tminus2_causal_prior"):
            rmse = 1.5
        else:
            rmse = 2.1
        data["results"][case] = {
            "solve": {
                "stabilized": True, "objective": 12.2,
                "passes": [{
                    "stabilized": True,
                    "position_shift_rmse_m": 0.001,
                    "relative_objective_reduction": 1e-7,
                }],
            },
            "evaluation": {k: rmse for k in a.METRICS},
        }
        data["case_sigmas"][case] = {
            "min_m": 1.0, "mean_xyz_m": [2.0, 3.0, 4.0],
            "max_m": 5.0,
        }
    return data


def test_v8_3_report_audit_all_frozen_methods():
    data = _fixture()
    summary = a.audit_report(data)
    assert summary["case_count"] == 16
    assert len(summary["metrics"]) == 16
    assert summary["all_stable"]
    assert summary["missing_h2_matched_baseline"]
    assert summary["best_3d_rmse_m"] == 1.5
    for model in a.MODELS:
        h1 = summary["comparisons"][model]["h1_tminus1_causal_prior"]
        assert h1["relative_gain_percent"] == pytest.approx(10.0)
        blend = summary["comparisons"][model]["h1h2_equal_causal_prior"]
        assert blend["relative_gain_percent"] == pytest.approx(-10.0)
        h2 = summary["comparisons"][model]["h2_tminus2_causal_prior"]
        assert h2["matched_h0_tminus2_fgo_baseline"] is None


def test_v8_3_report_audit_rejects_undocumented_methods():
    data = _fixture()
    data["results"]["gru96_r__h0_tminus2_persistence"] = copy.deepcopy(
        data["results"][a.CASES[1]])
    with pytest.raises(ValueError, match="exactly 16"):
        a.audit_report(data)
    data = _fixture()
    data["comparisons_frozen_before_dev_FGO"] = []
    with pytest.raises(ValueError, match="locked comparison"):
        a.audit_report(data)


def test_v8_3_report_audit_rejects_nonstable_or_nonconverged():
    data = _fixture()
    data["results"]["axis_fixed"]["solve"]["stabilized"] = False
    with pytest.raises(ValueError, match="not marked stable"):
        a.audit_report(data)
    data = _fixture()
    data["results"]["axis_fixed"]["solve"]["passes"][-1][
        "position_shift_rmse_m"] = 0.011
    with pytest.raises(ValueError, match="convergence gate"):
        a.audit_report(data)


def test_v8_3_report_audit_rejects_false_test_or_truth_claims():
    data = _fixture()
    data["heldout_0016_0020_accessed"] = True
    with pytest.raises(ValueError, match="provenance mismatch"):
        a.audit_report(data)
    data = _fixture()
    data["independent_position_ground_truth"] = True
    with pytest.raises(ValueError, match="provenance mismatch"):
        a.audit_report(data)


def test_v8_3_report_audit_rejects_invalid_sigmas():
    data = _fixture()
    data["case_sigmas"]["axis_fixed"]["max_m"] = 2.0
    with pytest.raises(ValueError, match="sigma mean outside"):
        a.audit_report(data)
    data = _fixture()
    data["case_sigmas"]["axis_fixed"]["min_m"] = -1.0
    with pytest.raises(ValueError, match="invalid numerical value"):
        a.audit_report(data)


def test_v8_3_report_audit_rejects_nonfinite_metrics():
    data = _fixture()
    data["results"]["axis_fixed"]["evaluation"][a.METRICS[0]] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        a.audit_report(data)


def test_v8_3_readonly_cli_never_writes_or_modifies_report(
        monkeypatch, tmp_path, capsys):
    path = tmp_path / "v8_dev_feedforward_fgo_results.json"
    path.write_text(json.dumps(_fixture()), encoding="utf-8")
    saved = path.read_bytes()
    timestamp = path.stat().st_mtime_ns
    monkeypatch.setattr(a, "REPORT_PATH", path)
    with pytest.raises(SystemExit):
        a.main([])
    with pytest.raises(SystemExit):
        a.main(["--run-graph"])
    a.main(["--audit"])
    captured = capsys.readouterr().out
    assert "V8_3_REPORT_READONLY_AUDIT_PASSED" in captured
    assert "NO MATCHED h0(t-2) FGO baseline" in captured
    assert path.read_bytes() == saved
    assert path.stat().st_mtime_ns == timestamp
    assert sorted(tmp_path.iterdir()) == [path]
