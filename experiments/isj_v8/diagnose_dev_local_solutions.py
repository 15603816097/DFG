from __future__ import annotations

"""V8.3.1: controlled same-objective, cross-initialization solver audit.

Only 0027 validation; no 0016/0020; no new model training; no feedback.
The two cases were chosen AFTER seeing V8.3 results, so this is an
EXPLORATORY numerical diagnosis, not a confirmatory method benchmark.
For each of TWO fixed GPS covariance objectives, start LM from:
(A) the original LiDAR initial trajectory and
(B) the converged solution of the other objective.
Cross-initialization NEVER changes the target objective's GPS weights.
The clean (correlated) GPS reference is read AFTER all solves, solely
for retrospective diagnostic RMSE.

Original frozen V8.3 JSON is read-only. This creates an independent,
new, one-time output file only after successful completion.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.isj_v7.cross_sequence_v7 import _positions, _solve_graph
from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, DEGRADED, END, reference_diagnostics, subgraph_measurements,
)
from experiments.isj_v8.audit_dev_fgo_results import (
    REPORT_PATH, audit_report,
)
from experiments.isj_v8.run_dev_feedforward_fgo import (
    prepare_cases,
)
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import (
    run_four_sensor_factor_graph,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/isj_v8/dev_0027_local_solution_probe_v1"
REPORT = OUT / "cross_initialization_diagnostics.json"
SOURCE = DEGRADED / "degraded_sensor_data.npz"
V8_3_ORIGINAL_SHA256 = (
    "4a716af6bbdd1f7edfcb9e3282191a781a06d013d8b574e9956c97a8446979ca"
)
CASE_GOOD = "mamba_r__h0_current_observation_available"
CASE_BAD = "mamba_r__h1h2_equal_causal_prior"
CASES = (CASE_GOOD, CASE_BAD)
TRIGGER_TOKEN = "V8_3_1_0027_DIAGNOSE_LOCAL_SOLUTIONS"
EXPECTED_COUNTS = {"gps": 680, "imu": 679, "lidar": 679, "camera": 670}
EXPECTED_3D_REFERENCE_TOL_M = 0.05
CROSS_OBJECTIVE_RELATIVE_THRESHOLD = 1e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sealed_source_report() -> tuple[dict, dict]:
    """Verify exact original V8.3 JSON before re-evaluating its objective."""
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"V8.3 frozen input is missing: {REPORT_PATH}")
    digest = _sha256(REPORT_PATH)
    if digest != V8_3_ORIGINAL_SHA256:
        raise ValueError("original V8.3 source JSON SHA256 mismatch")
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    summary = audit_report(data)
    if not summary["all_stable"]:
        raise ValueError("original V8.3 numerical stability failure")
    return data, summary


def source_preflight() -> tuple[dict, dict[str, np.ndarray], dict]:
    if OUT.exists():
        raise FileExistsError(
            f"Refuse to overwrite V8.3.1 output folder: {OUT}")
    frozen, audited = load_sealed_source_report()
    prepared, sigmas = prepare_cases()
    if (prepared.get("all_cases_use_same_frame_range") != [3180, 3860]
            or prepared.get("heldout_0016_0020_accessed") is not False):
        raise ValueError("V8.3.1 preparation changed development restrictions")
    # Frozen and freshly prepared measurement sources must coincide.
    for key in (
        "gps_degraded_sha256", "gps_candidate_sha256",
        "lidar_factors_sha256", "camera_factors_sha256",
    ):
        if frozen.get(key) != prepared.get(key):
            raise ValueError(f"V8.3.1 input hash mismatch: {key}")
    for case in CASES:
        if case not in sigmas or case not in audited["metrics"]:
            raise ValueError(f"missing locked V8.3 case: {case}")
        p = np.asarray(sigmas[case], dtype=np.float64)
        original = frozen["case_sigmas"][case]
        if (p.shape != (680, 3) or not np.isfinite(p).all()
                or np.any(p <= 0)
                or not math.isclose(float(p.min()), original["min_m"],
                                    rel_tol=0., abs_tol=1e-8)
                or not math.isclose(float(p.max()), original["max_m"],
                                    rel_tol=0., abs_tol=1e-8)
                or not np.allclose(
                    np.mean(p, axis=0), original["mean_xyz_m"],
                    rtol=0., atol=1e-8)):
            raise ValueError(f"{case}: GPS sigma is not frozen V8.3 mapping")
    metadata = {
        "schema": "v8_0027_dev_local_solutions_v1",
        "source": "2011_10_03_drive_0027_sync",
        "status": "PREFLIGHT_ONLY",
        "source_v8_3_json_sha256": V8_3_ORIGINAL_SHA256,
        "source_heldout_read": False,
        "new_heldout_read": False,
        "source_model_training": False,
        "independent_truth": False,
        "offline_batch_lm": True,
        "case_selection_posthoc_exploratory": True,
        "reference_influences_optimizer": False,
        "target_graph_objectives": list(CASES),
        "initialization_sources": [
            "common_lidar_initial",
            "other_weighting_converged_pose",
        ],
        "factors_per_graph": EXPECTED_COUNTS.copy(),
        "output_path": str(REPORT),
    }
    return metadata, {case: sigmas[case] for case in CASES}, frozen


def build_values_from_pose_matrices(
    gtsam, poses: np.ndarray,
):
    """Reconstitute Pose3 GTSAM Values with unmodified world-frame poses.

    Pose3(matrix4x4) is a supported GTSAM constructor. This must be a
    SE(3) transform; otherwise fail rather than silently project poses.
    """
    pose_array = np.asarray(poses, dtype=np.float64)
    if (pose_array.shape != (680, 4, 4)
            or not np.isfinite(pose_array).all()
            or not np.allclose(
                pose_array[:, 3, :],
                np.array([0., 0., 0., 1.]), rtol=0., atol=1e-7)
            or np.max(np.abs(
                np.matmul(
                    pose_array[:, :3, :3].transpose(0, 2, 1),
                    pose_array[:, :3, :3],
                ) - np.eye(3)
            )) > 1e-5
            or not np.allclose(
                np.linalg.det(pose_array[:, :3, :3]),
                1., rtol=0., atol=1e-5)):
        raise ValueError("invalid SE(3) optimized matrices; no seed conversion")
    values = gtsam.Values()
    for i, pose in enumerate(pose_array):
        values.insert(i, gtsam.Pose3(pose))
    return values


def compare_same_objective(
    default: dict, cross: dict, *, target_case: str,
) -> dict:
    """Within-graph comparison only. Never compare raw FGO objectives
    between DIFFERENT GPS covariance target cases.
    """
    a = float(default["objective"])
    b = float(cross["objective"])
    if (not np.isfinite([a, b]).all() or a <= 0 or b <= 0
            or default["stabilized"] is not True
            or cross["stabilized"] is not True):
        raise ValueError("solver result invalid or not numerically stable")
    relative = (b - a) / max(abs(a), 1.0)
    return {
        "target_case": target_case,
        "same_target_graph": True,
        "original_lidar_seed_objective": a,
        "cross_seed_objective": b,
        "cross_minus_default_objective": b-a,
        "cross_minus_default_relative_objective": relative,
        "distinct_stationary_objectives_threshold":
            CROSS_OBJECTIVE_RELATIVE_THRESHOLD,
        "distinct_stationary_objectives_detected":
            bool(abs(relative) > CROSS_OBJECTIVE_RELATIVE_THRESHOLD),
        "cross_seed_objective_is_lower": bool(
            relative < -CROSS_OBJECTIVE_RELATIVE_THRESHOLD),
        "interpretation": (
            "distinct stable LM objectives under identical factor weights"
            if abs(relative) > CROSS_OBJECTIVE_RELATIVE_THRESHOLD
            else "no meaningful objective separation at this threshold"
        ),
    }


def run_diagnostic() -> dict:
    metadata, sigmas, frozen = source_preflight()
    import gtsam

    measurements = load_degraded_four_sensor_measurements(DEGRADED)
    sub = subgraph_measurements(measurements, begin=BEGIN, end=END)
    graphs, initializations = {}, {}
    initial_matrices = None
    for target_case in CASES:
        graph, initial, counts = run_four_sensor_factor_graph(
            sub, OUT / "NO_WRITES", mode="fixed",
            gps_sigmas_override=sigmas[target_case],
            build_graph_only=True,
        )
        if counts != EXPECTED_COUNTS:
            raise ValueError(f"{target_case}: different sensor factors")
        mat = _positions(gtsam, initial, 680)
        if initial_matrices is None:
            initial_matrices = mat.copy()
        elif not np.array_equal(initial_matrices, mat):
            raise ValueError("different original LiDAR initial poses")
        graphs[target_case] = graph
        initializations[target_case] = initial

    default_results = {}
    for target_case in CASES:
        result = _solve_graph(
            graphs[target_case], initializations[target_case], 680)
        if not result["stabilized"]:
            raise RuntimeError(f"{target_case}: original seed not stabilized")
        default_results[target_case] = result
        print(
            "LOCAL_PROBE_DEFAULT_SOLVED", target_case,
            "objective", result["objective"],
            "passes", len(result["passes"]), flush=True,
        )

    cross_results = {}
    for target_case in CASES:
        other_case = CASE_BAD if target_case == CASE_GOOD else CASE_GOOD
        seed_values = build_values_from_pose_matrices(
            gtsam, default_results[other_case]["poses"])
        graph = graphs[target_case]
        initial_error = float(graph.error(seed_values))
        cross = _solve_graph(graph, seed_values, 680)
        if not cross["stabilized"]:
            raise RuntimeError(f"{target_case}: cross seed not stabilized")
        if (not math.isfinite(initial_error)
                or float(cross["objective"]) >
                initial_error + 1e-8 * max(1.0, abs(initial_error))):
            raise RuntimeError(f"{target_case}: cross solve invalid")
        cross_results[target_case] = {
            "initial_objective_in_target_graph": initial_error,
            "result": cross,
            "source": other_case,
        }
        print(
            "LOCAL_PROBE_CROSS_SOLVED", target_case,
            "seed", other_case,
            "objective", cross["objective"],
            "passes", len(cross["passes"]), flush=True,
        )

    # The OXTS/GPS-correlated reference is read only after all four solves.
    with np.load(SOURCE, allow_pickle=False) as src:
        reference = np.asarray(src["clean_gps_local"][BEGIN:END],
                               dtype=np.float64)
    if (reference.shape != (680, 3)
            or not np.isfinite(reference).all()):
        raise ValueError("invalid 0027 reference")
    metadata["solves"] = {}
    metric_key = "origin_aligned_oxTS_relative_3d_rmse_m"
    for case in CASES:
        base = default_results[case]
        cross = cross_results[case]["result"]
        base_eval = reference_diagnostics(base["poses"][:, :3, 3], reference)
        cross_eval = reference_diagnostics(cross["poses"][:, :3, 3], reference)
        original = frozen["results"][case]
        if not math.isclose(
            base_eval[metric_key], original["evaluation"][metric_key],
            rel_tol=0., abs_tol=EXPECTED_3D_REFERENCE_TOL_M):
            raise ValueError(f"{case}: failed to reproduce original DEV FGO")
        comparison = compare_same_objective(base, cross, target_case=case)
        metadata["solves"][case] = {
            "base_seed": {
                "source": "common_lidar_initial",
                "objective": float(base["objective"]),
                "passes": len(base["passes"]),
                "evaluation": base_eval,
            },
            "cross_seed": {
                "source": cross_results[case]["source"],
                "initial_objective_in_target_graph":
                    cross_results[case]["initial_objective_in_target_graph"],
                "objective": float(cross["objective"]),
                "passes": len(cross["passes"]),
                "evaluation": cross_eval,
            },
            "same_objective_comparison": comparison,
            "cross_minus_default_3d_rmse_m":
                cross_eval[metric_key]-base_eval[metric_key],
        }

    metadata["status"] = "COMPLETE_0027_DEV_EXPLORATORY_LOCAL_PROBE"
    metadata["source_reference_is_correlated"] = True
    OUT.mkdir(parents=True, exist_ok=False)
    REPORT.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": metadata["status"],
        "cases": metadata["solves"],
        "report_path": str(REPORT),
        "result_sha256": _sha256(REPORT),
    }, indent=2, ensure_ascii=False), flush=True)
    print(
        "V8_DEV_LOCAL_SOLUTION_PROBE_COMPLETE; 0027_DEV_ONLY; "
        "4_SOLVES; NO_HELDOUT; NO_RETRAIN",
        flush=True,
    )
    return metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run-probe", action="store_true")
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)
    if args.preflight:
        if args.token is not None:
            parser.error("preflight does not accept token")
        metadata, _, _ = source_preflight()
        print(json.dumps(metadata, indent=2), flush=True)
        print(
            "V8_DEV_LOCAL_PROBE_PREFLIGHT_OK; NO_SOLVES; NO_WRITES; "
            "NO_HELDOUT",
            flush=True,
        )
        return
    if args.token != TRIGGER_TOKEN:
        parser.error("explicit development-only local-solution token required")
    run_diagnostic()


if __name__ == "__main__":
    main()
