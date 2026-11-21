from __future__ import annotations

"""Read-only V8.3 report audit: no GTSAM, dataset, model or solver access.

Prints all 16 already-solved 0027 development graph results. H2 is
reported descriptively because the h0(t-2) matched FGO baseline was
NOT in the locked 16-case protocol. Never fabricate that missing run.
All metrics reference GPS-correlated OXTS, NOT independent physical GT.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = (
    ROOT / "results/isj_v8/dev_0027_feedforward_fgo_v1"
    / "v8_dev_feedforward_fgo_results.json"
)
MODELS = ("gru96_r", "gru106_r", "mamba_r")
METHODS = (
    "h0_current_observation_available",
    "h0_tminus1_persistence",
    "h1_tminus1_causal_prior",
    "h2_tminus2_causal_prior",
    "h1h2_equal_causal_prior",
)
CASES = ("axis_fixed",) + tuple(
    f"{model}__{method}" for model in MODELS for method in METHODS
)
METRICS = (
    "origin_aligned_oxTS_relative_3d_rmse_m",
    "origin_aligned_oxTS_relative_2d_rmse_m",
    "relative_displacement_1_frame_rmse_m",
    "relative_displacement_10_frame_rmse_m",
)
FACTORS = {"gps": 680, "imu": 679, "lidar": 679, "camera": 670}


def _number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a finite numerical value")
    v = float(value)
    if not math.isfinite(v) or v < 0 or (positive and v <= 0):
        raise ValueError(f"{name}: nonfinite or invalid numerical value")
    return v


def audit_report(data: dict) -> dict:
    """Validate immutable V8.3 report structure and compute valid comparisons.

    This function cannot modify a result file; it accepts only in-memory
    data and returns a new summary.
    """
    if not isinstance(data, dict):
        raise ValueError("unexpected report format")
    expected = {
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
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"V8.3 report provenance mismatch: {key}")
    if data.get("comparisons_frozen_before_dev_FGO") != list(CASES):
        raise ValueError("V8.3 locked comparison cases changed")
    if data.get("factors_per_graph") != FACTORS:
        raise ValueError("V8.3 factor counts mismatch")

    cases = data.get("results")
    sigmas = data.get("case_sigmas")
    if (not isinstance(cases, dict) or set(cases) != set(CASES)
            or not isinstance(sigmas, dict)
            or set(sigmas) != set(CASES)):
        raise ValueError("V8.3 must contain exactly 16 locked cases")
    metrics: dict[str, dict[str, float]] = {}
    solver: dict[str, dict] = {}
    for case in CASES:
        entry = cases[case]
        if not isinstance(entry, dict):
            raise ValueError(f"{case}: malformed graph result")
        solve = entry.get("solve", {})
        passes = solve.get("passes")
        if (solve.get("stabilized") is not True
                or not isinstance(passes, list) or not passes
                or passes[-1].get("stabilized") is not True):
            raise ValueError(f"{case}: solver is not marked stable")
        objective = _number(solve.get("objective"), f"{case}: objective")
        shift = _number(
            passes[-1].get("position_shift_rmse_m"),
            f"{case}: final position shift",
        )
        improvement = _number(
            passes[-1].get("relative_objective_reduction"),
            f"{case}: final relative objective reduction",
        )
        if shift > 0.01000000001 or improvement > 0.000001000000001:
            raise ValueError(f"{case}: final pass exceeds convergence gate")
        evaluation = entry.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError(f"{case}: missing evaluation")
        metrics[case] = {
            k: _number(evaluation.get(k), f"{case}:{k}")
            for k in METRICS
        }
        axis = sigmas[case]
        sigma_min = _number(
            axis.get("min_m"), f"{case}: sigma min", positive=True)
        sigma_max = _number(
            axis.get("max_m"), f"{case}: sigma max", positive=True)
        mean = axis.get("mean_xyz_m")
        if (not isinstance(mean, list) or len(mean) != 3
                or sigma_max < sigma_min):
            raise ValueError(f"{case}: invalid sigma metadata")
        for value in mean:
            v = _number(value, f"{case}: sigma mean", positive=True)
            if v < sigma_min-1e-8 or v > sigma_max+1e-8:
                raise ValueError(f"{case}: sigma mean outside min/max")
        solver[case] = {
            "lm_passes": len(passes),
            "final_graph_objective": objective,
            "final_position_shift_rmse_m": shift,
            "final_relative_objective_reduction": improvement,
        }

    primary = METRICS[0]
    comparisons = {}
    for model in MODELS:
        base = metrics[f"{model}__h0_tminus1_persistence"][primary]
        model_results = {}
        for alternative in (
            "h1_tminus1_causal_prior",
            "h1h2_equal_causal_prior",
        ):
            value = metrics[f"{model}__{alternative}"][primary]
            gain = base - value
            model_results[alternative] = {
                "baseline_h0_tminus1_rmse_m": base,
                "forecast_rmse_m": value,
                "absolute_gain_m": gain,
                "relative_gain_percent": 100 * gain / base,
            }
        model_results["h2_tminus2_causal_prior"] = {
            "forecast_rmse_m": metrics[
                f"{model}__h2_tminus2_causal_prior"][primary],
            "matched_h0_tminus2_fgo_baseline": None,
            "comparability": "NO_MATCHED_TWO_FRAME_BASELINE",
        }
        comparisons[model] = model_results

    best = min(CASES, key=lambda c: metrics[c][primary])
    return {
        "case_count": len(CASES),
        "all_stable": True,
        "metrics": metrics,
        "solver": solver,
        "comparisons": comparisons,
        "best_case": best,
        "best_3d_rmse_m": metrics[best][primary],
        "missing_h2_matched_baseline": True,
        "no_independent_truth": True,
        "not_online_fgo": True,
    }


def print_readonly_summary(summary: dict, sha256: str) -> None:
    print("V8.3 DEVELOPMENT FGO: ALL 16 FROZEN RESULTS")
    print(
        f"{'CASE':52s} {'3D[m]':>10s} {'2D[m]':>10s}"
        f" {'R1[m]':>10s} {'R10[m]':>10s} {'LM':>3s}"
    )
    for case in CASES:
        entry = summary["metrics"][case]
        print(
            f"{case:52s}"
            + "".join(f" {entry[k]:10.6f}" for k in METRICS)
            + f" {summary['solver'][case]['lm_passes']:3d}"
        )
    print("\nVALID MATCHED 3D RMSE COMPARISONS (0027 DEV ONLY)")
    for model in MODELS:
        print(f"MODEL={model}")
        for method in (
            "h1_tminus1_causal_prior", "h1h2_equal_causal_prior"
        ):
            row = summary["comparisons"][model][method]
            print(
                f"  {method}: baseline={row['baseline_h0_tminus1_rmse_m']:.6f} m"
                f", forecast={row['forecast_rmse_m']:.6f} m"
                f", gain={row['relative_gain_percent']:+.3f}%"
            )
        h2 = summary["comparisons"][model]["h2_tminus2_causal_prior"]
        print(
            "  h2(t-2): 3D RMSE="
            f"{h2['forecast_rmse_m']:.6f} m; "
            "NO MATCHED h0(t-2) FGO baseline; no gain reported"
        )
    print("\nBEST_3D_METHOD:", summary["best_case"])
    print(f"BEST_3D_RMSE_M: {summary['best_3d_rmse_m']:.6f}")
    print("ALL_NUMERICAL_STABILITY_FLAGS: True")
    print("FACTORS_PER_GRAPH:", FACTORS, "(plus initial pose prior)")
    print("JSON_SHA256:", sha256)
    print("SCIENCE_LIMITS: 0027 dev; OXTS/GPS correlated reference; "
          "offline batch LM; one seed; no feedback.")
    print("OPTIMIZER_WARNING: different covariance weights imply different "
          "objectives; do not compare objective values across cases. "
          "Stable != globally optimal. Distinct error clusters warrant "
          "separate local-minima/weight-sensitivity study.")
    print("V8_3_REPORT_READONLY_AUDIT_PASSED")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", action="store_true", required=True,
        help="read only the already-saved V8.3 0027 development JSON",
    )
    parser.parse_args(argv)
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(REPORT_PATH)
    raw = REPORT_PATH.read_bytes()
    data = json.loads(raw)
    summary = audit_report(data)
    sha = hashlib.sha256(raw).hexdigest()
    print_readonly_summary(summary, sha)


if __name__ == "__main__":
    main()
