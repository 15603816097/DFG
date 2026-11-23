from __future__ import annotations

"""V8.3.2: verify FIRST-ORDER stationarity of the V8.3.1 four solutions.

EXPLORATORY 0027 DEV only. Identical fixed covariance graphs and seeds
as V8.3.1, recomputed only because the V8.3.1 archive retains summary
metrics, not the 680 optimized Pose3 matrices. Never accesses OXTS clean
reference, reserved 0016/0020, model training, or original result writes.
No new optimizer parameters, stopping-rule tuning, or feedback models.

The critical question is whether LM's "stable=True" combined with
zero actual progress is genuine first-order stationarity, or a false
convergence conclusion from the external stop criterion.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.isj_v7.cross_sequence_v7 import _positions, _solve_graph
from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, DEGRADED, END, subgraph_measurements,
)
from experiments.isj_v8.diagnose_dev_local_solutions import (
    CASES, CASE_BAD, CASE_GOOD, EXPECTED_COUNTS,
    V8_3_ORIGINAL_SHA256, build_values_from_pose_matrices,
    load_sealed_source_report,
)
from experiments.isj_v8.run_dev_feedforward_fgo import prepare_cases
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/isj_v8/dev_0027_stationarity_audit_v1"
REPORT = OUT / "stationarity_results.json"
PREVIOUS_REPORT = (
    ROOT / "results/isj_v8/dev_0027_local_solution_probe_v1"
    / "cross_initialization_diagnostics.json"
)
EXPECTED_PREVIOUS_SHA256 = (
    "c684238ad7b97c23add7e116129ab4b2d9be8ad79fe7017d2c2e3140dca3a854"
)
TOKEN = "V8_3_2_0027_STATIONARITY_DIAGNOSTIC"
N = END - BEGIN
POSE_DOF = 6
# Strict reproducibility tolerance comparing recorded objective to fresh solve.
REL_REPLAY_TOL = 1e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_existing_probes() -> tuple[dict, dict[str, np.ndarray]]:
    if OUT.exists():
        raise FileExistsError(f"V8.3.2 destination exists; refuse overwrite: {OUT}")
    frozen, _ = load_sealed_source_report()
    if sha(PREVIOUS_REPORT) != EXPECTED_PREVIOUS_SHA256:
        raise ValueError("V8.3.1 archived result SHA256 mismatch")
    prior = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    if (prior.get("status") != "COMPLETE_0027_DEV_EXPLORATORY_LOCAL_PROBE"
            or prior.get("source_v8_3_json_sha256") != V8_3_ORIGINAL_SHA256
            or prior.get("new_heldout_read") is not False
            or set(prior.get("solves", {})) != set(CASES)):
        raise ValueError("V8.3.1 results are not the exact frozen DEV probe")
    fresh, sigmas = prepare_cases()
    if (fresh.get("all_cases_use_same_frame_range") != [3180, 3860]
            or fresh.get("heldout_0016_0020_accessed") is not False):
        raise ValueError("V8.3.2 data restriction mismatch")
    for name in (
        "gps_degraded_sha256", "gps_candidate_sha256",
        "lidar_factors_sha256", "camera_factors_sha256",
    ):
        if fresh.get(name) != frozen.get(name):
            raise ValueError(f"V8.3.2 source hash mismatch: {name}")
    for case in CASES:
        a = np.asarray(sigmas[case], dtype=np.float64)
        old = frozen["case_sigmas"][case]
        if (a.shape != (N, 3) or not np.isfinite(a).all()
                or np.any(a <= 0)
                or not math.isclose(float(a.min()), old["min_m"],
                                    rel_tol=0., abs_tol=1e-8)
                or not math.isclose(float(a.max()), old["max_m"],
                                    rel_tol=0., abs_tol=1e-8)
                or not np.allclose(np.mean(a, axis=0),
                                   old["mean_xyz_m"], rtol=0, atol=1e-8)):
            raise ValueError("original GPS sigma mapping changed")
    return prior, {case: sigmas[case] for case in CASES}


def gradient_norm(graph, state) -> float:
    """GTSAM J^T r at one Pose3 Values estimate.

    graph.linearize(Values).gradientAtZero().norm() has GTSAM API
    documentation. The quantity is a mixed-rotation/translation tangent
    coordinate norm: no universal physical stationarity threshold.
    """
    try:
        norm = float(graph.linearize(state).gradientAtZero().norm())
    except Exception as exc:
        raise RuntimeError(
            "GTSAM gradientAtZero unsupported or graph non-linearizable; "
            "do not infer first-order convergence"
        ) from exc
    if not math.isfinite(norm) or norm < 0:
        raise RuntimeError("invalid GTSAM stationarity gradient norm")
    return norm


def inspect_result(gtsam, graph, optimized: dict) -> dict:
    """Objective/gradient consistency for the exact supplied solution."""
    poses = np.asarray(optimized["poses"], dtype=np.float64)
    values = build_values_from_pose_matrices(gtsam, poses)
    objective = float(graph.error(values))
    saved = float(optimized["objective"])
    if (not math.isfinite(objective) or objective < 0
            or not math.isclose(objective, saved,
                                rel_tol=REL_REPLAY_TOL, abs_tol=1e-6)
            or optimized.get("stabilized") is not True):
        raise ValueError("reconstructed terminal GTSAM Values != solver output")
    gnorm = gradient_norm(graph, values)
    return {
        "objective": objective,
        "gradient_l2": gnorm,
        "gradient_l2_per_sqrt_tangent_dimension":
            gnorm / math.sqrt(N * POSE_DOF),
        "optimizer_marked_stable": True,
        "lm_passes": len(optimized["passes"]),
        "last_pass": optimized["passes"][-1],
    }


def run_diagnostic() -> dict:
    previous, sigmas = validate_existing_probes()
    import gtsam
    measurement = load_degraded_four_sensor_measurements(DEGRADED)
    sub = subgraph_measurements(measurement, begin=BEGIN, end=END)
    graphs = {}
    initials = {}
    original_initial_poses = None
    for case in CASES:
        graph, initial, factor_counts = run_four_sensor_factor_graph(
            sub, OUT / "NO_GRAPH_WRITES", mode="fixed",
            gps_sigmas_override=sigmas[case], build_graph_only=True,
        )
        if factor_counts != EXPECTED_COUNTS:
            raise ValueError("changed V8.3 sensor factor counts")
        poses = _positions(gtsam, initial, N)
        if original_initial_poses is None:
            original_initial_poses = poses.copy()
        elif not np.array_equal(poses, original_initial_poses):
            raise ValueError("original LiDAR seeds differ across target graphs")
        graphs[case], initials[case] = graph, initial

    result = {
        "schema": "v8_0027_dev_stationarity_v1",
        "status": "IN_PROGRESS",
        "development_only": True,
        "posthoc_diagnostic": True,
        "no_reference_or_oracle_loaded_for_evaluation": True,
        "heldout_0016_0020_accessed": False,
        "same_target_objective_comparisons_only": True,
        "source_v8_3_sha256": V8_3_ORIGINAL_SHA256,
        "source_v8_3_1_sha256": EXPECTED_PREVIOUS_SHA256,
        "source_covariance_mapping": "unchanged V8.3 0027 train-only Cap4",
        "gradient_definition": "||J^T r||_2 in GTSAM tangent coordinates",
        "gradient_threshold_declared": False,
        "read_only_original_result_files": True,
        "case_diagnostics": {},
    }
    defaults = {}
    for case in CASES:
        g = graphs[case]
        initial_1st_order = gradient_norm(g, initials[case])
        objective_init = float(g.error(initials[case]))
        solved = _solve_graph(g, initials[case], N)
        info = inspect_result(gtsam, g, solved)
        archived = previous["solves"][case]["base_seed"]["objective"]
        if not math.isclose(info["objective"], archived,
                            rel_tol=REL_REPLAY_TOL, abs_tol=1e-6):
            raise ValueError(f"{case}: default solver objective not reproducible")
        defaults[case] = solved
        result["case_diagnostics"][case] = {
            "common_lidar_initial": {
                "initial_objective": objective_init,
                "initial_gradient_l2": initial_1st_order,
                "solved": info,
                "objective_drop": objective_init - info["objective"],
            },
        }
        print("STATIONARITY_DEFAULT", case,
              "gradient_norm", info["gradient_l2"],
              "objective", info["objective"], flush=True)

    for case in CASES:
        other = CASE_BAD if case == CASE_GOOD else CASE_GOOD
        g = graphs[case]
        seed = build_values_from_pose_matrices(
            gtsam, defaults[other]["poses"])
        error_start = float(g.error(seed))
        norm_start = gradient_norm(g, seed)
        solved = _solve_graph(g, seed, N)
        info = inspect_result(gtsam, g, solved)
        archived = previous["solves"][case]["cross_seed"]["objective"]
        if not math.isclose(info["objective"], archived,
                            rel_tol=REL_REPLAY_TOL, abs_tol=1e-6):
            raise ValueError(f"{case}: cross solver objective not reproducible")
        result["case_diagnostics"][case]["other_solution_initial"] = {
            "seed_source_case": other,
            "initial_objective": error_start,
            "initial_gradient_l2": norm_start,
            "solved": info,
            "objective_drop": error_start - info["objective"],
            "no_objective_progress": math.isclose(
                error_start, info["objective"], rel_tol=1e-10, abs_tol=1e-8),
        }
        print("STATIONARITY_CROSS", case,
              "initial_gradient", norm_start,
              "terminal_gradient", info["gradient_l2"],
              "objective", info["objective"], flush=True)

    result["status"] = "COMPLETE_0027_DEV_STATIONARITY_DIAGNOSTIC"
    OUT.mkdir(parents=True, exist_ok=False)
    REPORT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "case_diagnostics": result["case_diagnostics"],
        "output_sha256": sha(REPORT),
    }, ensure_ascii=False, indent=2), flush=True)
    print("V8_DEV_STATIONARITY_DIAGNOSTIC_COMPLETE; "
          "4_SOLVES; NO_REFERENCE; NO_HELDOUT", flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-diagnostic", action="store_true")
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)
    if args.preflight:
        if args.token is not None:
            parser.error("preflight cannot use token")
        _, sigmas = validate_existing_probes()
        print(json.dumps({
            "schema": "v8_0027_dev_stationarity_preflight_v1",
            "source_v8_3_1_sha256": EXPECTED_PREVIOUS_SHA256,
            "target_cases": list(sigmas),
            "no_solver_or_writes": True,
            "heldout_read": False,
        }, indent=2), flush=True)
        print("V8_DEV_STATIONARITY_PREFLIGHT_OK; "
              "NO_SOLVER; NO_WRITES; NO_HELDOUT", flush=True)
        return
    if args.token != TOKEN:
        parser.error("explicit V8.3.2 DEV-only token required")
    run_diagnostic()


if __name__ == "__main__":
    main()
