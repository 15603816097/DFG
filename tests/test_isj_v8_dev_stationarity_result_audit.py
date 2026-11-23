from __future__ import annotations

import copy
import json

import pytest

from experiments.isj_v8 import audit_dev_stationarity_results as audit


def _fixture():
    raw = {
        "schema": "v8_0027_dev_stationarity_v1",
        "status": "COMPLETE_0027_DEV_STATIONARITY_DIAGNOSTIC",
        "development_only": True,
        "posthoc_diagnostic": True,
        "no_reference_or_oracle_loaded_for_evaluation": True,
        "heldout_0016_0020_accessed": False,
        "same_target_objective_comparisons_only": True,
        "source_v8_3_sha256": audit.SOURCE_V8_3_SHA,
        "source_v8_3_1_sha256": audit.SOURCE_V8_3_1_SHA,
        "gradient_threshold_declared": False,
        "read_only_original_result_files": True,
        "case_diagnostics": {},
    }
    for case in audit.CASES:
        is_current = case == audit.CASE_CURRENT
        first_objective = 2044.7 if is_current else 10967.3
        second_objective = 11090.5 if is_current else 2236.6
        g_first = 4.59 if is_current else 263.925
        g_second = 274.441 if is_current else 2.096
        first_delta = 0.001 if is_current else 0.0
        first_shift = 0.01 if is_current else 0.0
        second_delta = 0.0 if is_current else 0.001
        second_shift = 0.0 if is_current else 0.01
        raw["case_diagnostics"][case] = {
            "common_lidar_initial": {
                "initial_gradient_l2": 2000.,
                "solved": {
                    "objective": first_objective, "gradient_l2": g_first,
                    "gradient_l2_per_sqrt_tangent_dimension":
                        g_first / (4080 ** .5),
                    "optimizer_marked_stable": True,
                    "lm_passes": 2,
                    "last_pass": {
                        "pass": 2, "stabilized": True,
                        "objective_after": first_objective,
                        "relative_objective_reduction": first_delta,
                        "position_shift_rmse_m": first_shift,
                    },
                },
            },
            "other_solution_initial": {
                "initial_objective": (
                    second_objective if is_current else second_objective+55),
                "initial_gradient_l2": (g_second if is_current else 20.),
                "no_objective_progress": is_current,
                "solved": {
                    "objective": second_objective, "gradient_l2": g_second,
                    "gradient_l2_per_sqrt_tangent_dimension":
                        g_second / (4080 ** .5),
                    "optimizer_marked_stable": True,
                    "lm_passes": 1,
                    "last_pass": {
                        "pass": 1, "stabilized": True,
                        "objective_after": second_objective,
                        "relative_objective_reduction": second_delta,
                        "position_shift_rmse_m": second_shift,
                    },
                },
            },
        }
    return raw


def test_v8_stationarity_status_distinguishes_stall_vs_nonstall():
    summary = audit.audit_stationarity_report(_fixture())
    assert set(summary["cases"]) == set(audit.CASES)
    case = summary["cases"][audit.CASE_CURRENT]["seeds"]
    assert case["other_solution_initial"]["last_step_no_progress"] is True
    assert case["other_solution_initial"]["gradient_unchanged_since_seed"] is True
    assert case["common_lidar_initial"]["last_step_no_progress"] is False
    other = summary["cases"][audit.CASE_BLEND]["seeds"]
    assert other["common_lidar_initial"]["last_step_no_progress"] is True
    assert other["other_solution_initial"]["last_step_no_progress"] is False
    assert other["common_lidar_initial"]["classification"].startswith(
        "STALLED")
    assert summary["reliable_fgo_performance_conclusion"] is False


def test_v8_stationarity_status_accepts_no_global_threshold():
    summary = audit.audit_stationarity_report(_fixture())
    for case in audit.CASES:
        r = summary["cases"][case]
        assert r["no_cross_objective_comparison"] is True
        assert r["ratio_of_larger_to_smaller_terminal_gradients"] > 1.0


def test_v8_stationarity_status_rejects_wrong_source_hash():
    bad = _fixture()
    bad["source_v8_3_sha256"] = "TAMPERED"
    with pytest.raises(ValueError, match="source_v8_3_sha256"):
        audit.audit_stationarity_report(bad)


def test_v8_stationarity_status_rejects_extra_case():
    bad = _fixture()
    bad["case_diagnostics"]["unexpected"] = {}
    with pytest.raises(ValueError, match="exactly the two"):
        audit.audit_stationarity_report(bad)


def test_v8_stationarity_status_rejects_inconsistent_last_objective():
    bad = _fixture()
    bad["case_diagnostics"][audit.CASE_CURRENT][
        "common_lidar_initial"]["solved"]["last_pass"]["objective_after"] = 99.
    with pytest.raises(ValueError, match="last pass objective"):
        audit.audit_stationarity_report(bad)


def test_v8_stationarity_status_rejects_inconsistent_norm_scaling():
    bad = _fixture()
    bad["case_diagnostics"][audit.CASE_CURRENT][
        "common_lidar_initial"]["solved"][
        "gradient_l2_per_sqrt_tangent_dimension"] = 1.0
    with pytest.raises(ValueError, match="tangent-dimension"):
        audit.audit_stationarity_report(bad)


def test_v8_stationarity_status_rejects_fabricated_progress_flag():
    bad = _fixture()
    bad["case_diagnostics"][audit.CASE_CURRENT][
        "other_solution_initial"]["no_objective_progress"] = False
    with pytest.raises(ValueError, match="no-progress"):
        audit.audit_stationarity_report(bad)


def test_v8_stationarity_status_cli_is_readonly(monkeypatch, tmp_path, capsys):
    import hashlib
    p = tmp_path / "result.json"
    p.write_text(json.dumps(_fixture()), encoding="utf-8")
    before = p.read_bytes()
    timestamp = p.stat().st_mtime_ns
    monkeypatch.setattr(audit, "RESULT_PATH", p)
    monkeypatch.setattr(
        audit, "RESULT_SHA256", hashlib.sha256(before).hexdigest())
    with pytest.raises(SystemExit):
        audit.main([])
    with pytest.raises(SystemExit):
        audit.main(["--run-graph"])
    audit.main(["--audit"])
    output = capsys.readouterr().out
    assert "V8_3_2_STATIONARITY_READONLY_AUDIT_PASSED" in output
    assert p.read_bytes() == before
    assert p.stat().st_mtime_ns == timestamp
