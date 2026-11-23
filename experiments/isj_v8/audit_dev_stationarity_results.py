from __future__ import annotations

"""Read-only audit of the frozen V8.3.2 DEV gradient/stationarity result.

Explicitly does NOT claim to prove whether solutions are local minima.
Checks original V8.3.2 JSON hash and correct four-solution pairing.
Diagnoses zero objective/pose progress despite residual gradient and
reports within-SAME-objective gradient differences. Never runs GTSAM,
opens sensor datasets, recalibrates weights or writes outputs.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = (
    ROOT / "results/isj_v8/dev_0027_stationarity_audit_v1"
    / "stationarity_results.json"
)
RESULT_SHA256 = (
    "9f690dff405aeb08fbef5bb7cd68b6afe52e913649ff79832566293cb5cea9da"
)
SOURCE_V8_3_SHA = (
    "4a716af6bbdd1f7edfcb9e3282191a781a06d013d8b574e9956c97a8446979ca"
)
SOURCE_V8_3_1_SHA = (
    "c684238ad7b97c23add7e116129ab4b2d9be8ad79fe7017d2c2e3140dca3a854"
)
CASE_CURRENT = "mamba_r__h0_current_observation_available"
CASE_BLEND = "mamba_r__h1h2_equal_causal_prior"
CASES = (CASE_CURRENT, CASE_BLEND)
SEEDS = ("common_lidar_initial", "other_solution_initial")


def _value(x, key: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{key}: invalid numeric type")
    f = float(x)
    if not math.isfinite(f) or f < 0:
        raise ValueError(f"{key}: nonfinite or negative numeric value")
    return f


def audit_stationarity_report(raw: dict) -> dict:
    """Audit documented cases and terminal gradients; no solver access."""
    required = {
        "schema": "v8_0027_dev_stationarity_v1",
        "status": "COMPLETE_0027_DEV_STATIONARITY_DIAGNOSTIC",
        "development_only": True,
        "posthoc_diagnostic": True,
        "no_reference_or_oracle_loaded_for_evaluation": True,
        "heldout_0016_0020_accessed": False,
        "same_target_objective_comparisons_only": True,
        "source_v8_3_sha256": SOURCE_V8_3_SHA,
        "source_v8_3_1_sha256": SOURCE_V8_3_1_SHA,
        "gradient_threshold_declared": False,
        "read_only_original_result_files": True,
    }
    if not isinstance(raw, dict):
        raise ValueError("V8.3.2 report must be a JSON object")
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise ValueError(f"V8.3.2 source schema mismatch: {key}")
    diagnostics = raw.get("case_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(CASES):
        raise ValueError("V8.3.2 must contain exactly the two original targets")

    summary = {
        "schema": "v8_0027_dev_stationarity_readonly_audit_v1",
        "read_only": True,
        "independent_evaluation": False,
        "reliable_fgo_performance_conclusion": False,
        "source_sha256": RESULT_SHA256,
        "cases": {},
    }
    for case in CASES:
        d = diagnostics[case]
        if not isinstance(d, dict) or set(d) != set(SEEDS):
            raise ValueError(f"{case}: missing frozen initialization")
        rows = {}
        for seed in SEEDS:
            block = d[seed]
            solved = block.get("solved")
            if not isinstance(solved, dict) or solved.get("optimizer_marked_stable") is not True:
                raise ValueError(f"{case}/{seed}: missing solver stable flag")
            obj = _value(solved.get("objective"), "objective")
            gradient = _value(solved.get("gradient_l2"), "gradient_l2")
            gscale = _value(
                solved.get("gradient_l2_per_sqrt_tangent_dimension"),
                "scaled gradient",
            )
            if not math.isclose(
                gscale, gradient / math.sqrt(680 * 6), rel_tol=1e-10, abs_tol=1e-9
            ):
                raise ValueError("invalid tangent-dimension normalization")
            last = solved.get("last_pass")
            if not isinstance(last, dict) or last.get("stabilized") is not True:
                raise ValueError("missing last-pass record")
            delta = _value(
                last.get("relative_objective_reduction"), "last relative drop"
            )
            shift = _value(
                last.get("position_shift_rmse_m"), "last pose shift"
            )
            if not math.isclose(
                _value(last.get("objective_after"), "last objective"),
                obj, rel_tol=1e-10, abs_tol=1e-8
            ):
                raise ValueError("last pass objective != terminal objective")
            if last.get("pass") != solved.get("lm_passes"):
                raise ValueError("LM pass count inconsistent")
            stagnated = (delta == 0.0 and shift == 0.0)
            if seed == "other_solution_initial":
                initial_grad = _value(
                    block.get("initial_gradient_l2"), "initial gradient"
                )
                final_grad_retained = (
                    math.isclose(gradient, initial_grad,
                                 rel_tol=1e-10, abs_tol=1e-8)
                )
                expected_no_progress = block.get("no_objective_progress")
                if not isinstance(expected_no_progress, bool):
                    raise ValueError("missing no-progress classification")
                if expected_no_progress != math.isclose(
                    _value(block.get("initial_objective"), "initial objective"),
                    obj, rel_tol=1e-10, abs_tol=1e-8
                ):
                    raise ValueError("inconsistent cross initial no-progress flag")
            else:
                initial_grad = _value(
                    block.get("initial_gradient_l2"), "initial gradient"
                )
                final_grad_retained = math.isclose(
                    gradient, initial_grad, rel_tol=1e-10, abs_tol=1e-8
                )
            rows[seed] = {
                "objective": obj,
                "gradient_l2": gradient,
                "gradient_l2_per_sqrt_4080_dof": gscale,
                "initial_gradient_l2": initial_grad,
                "last_step_no_progress": stagnated,
                "gradient_unchanged_since_seed": final_grad_retained,
                "classification": (
                    "STALLED_LAST_PASS_WITH_NONZERO_GRADIENT"
                    if stagnated and gradient > 0.0
                    else "PROGRESS_RECORDED_BUT_STATIONARITY_NOT_CERTIFIED"
                ),
            }
        g1 = rows["common_lidar_initial"]["gradient_l2"]
        g2 = rows["other_solution_initial"]["gradient_l2"]
        summary["cases"][case] = {
            "seeds": rows,
            "ratio_of_larger_to_smaller_terminal_gradients":
                max(g1, g2)/min(g1, g2) if min(g1, g2) > 0.0 else None,
            "lower_gradient_seed": (
                "common_lidar_initial"
                if g1 < g2 else "other_solution_initial"
            ),
            "no_cross_objective_comparison": True,
        }
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args(argv)
    if not RESULT_PATH.is_file():
        raise FileNotFoundError(RESULT_PATH)
    raw_bytes = RESULT_PATH.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != RESULT_SHA256:
        raise ValueError("V8.3.2 frozen stationarity JSON SHA256 mismatch")
    result = audit_stationarity_report(json.loads(raw_bytes))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(
        "V8_3_2_STATIONARITY_READONLY_AUDIT_PASSED; "
        "4_SOLUTIONS; NO_WRITES; NO_SOLVER; NO_HELDOUT",
        flush=True,
    )


if __name__ == "__main__":
    main()
