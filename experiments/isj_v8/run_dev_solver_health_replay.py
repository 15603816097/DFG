from __future__ import annotations

"""V8.3.3 exploratory solver-health replay; frozen 0027 DEV only.

Goal: separate numerical-stall failures from covariance-model effects.
All 16 V8.3 fixed target graphs, the exact original Cap4 noise mapping,
the same 680 frames, same factors and identical LM inner parameters.

Two *reference-independent* common seeds for ALL target graphs:
  lidar_seed: original LiDAR-odometry initial pose trajectory.
  degraded_gps_translation_lidar_rotation: pose rotations from the
      same LiDAR seed but translations from the DEGRADED measured GPS.

For EACH target graph independently select minimum objective from
CONVERGED_CANDIDATE outcomes only; never use (correlated) reference
RMSE for initial values, solver settings, or candidate selection.

Uses an improved V8-specific STATUS wrapper, not an alternate
optimizer or a proven global minimum. Source/experiment are posthoc
and heavily explored; results remain development-only and do not
authorize any new 0016/0020 tuning or reruns.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.isj_v7.cross_sequence_v7 import _positions
from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, DEGRADED, END, reference_diagnostics, subgraph_measurements,
)
from experiments.isj_v8.audit_dev_fgo_results import (
    CASES, FACTORS, audit_report,
)
from experiments.isj_v8.diagnose_dev_local_solutions import (
    build_values_from_pose_matrices, load_sealed_source_report,
)
from experiments.isj_v8.diagnose_dev_stationarity import (
    EXPECTED_PREVIOUS_SHA256,
)
from experiments.isj_v8.run_dev_feedforward_fgo import (
    prepare_cases,
)
from experiments.isj_v8.solver_health_v8 import (
    select_reliable_candidate, solve_v8_health,
)
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/isj_v8/dev_0027_solver_health_replay_v1"
METADATA = OUT / "v8_dev_solver_health_results.json"
POSE_ARCHIVE = OUT / "all_solver_poses.npz"
SOURCE = DEGRADED / "degraded_sensor_data.npz"
SEED_NAMES = (
    "common_lidar_odometry",
    "degraded_gps_translation_lidar_rotation",
)
TOKEN = "V8_DEV_0027_SOLVER_HEALTH_REPLAY"
EXPECTED_V8_3_STATIONARITY_SHA = (
    "9f690dff405aeb08fbef5bb7cd68b6afe52e913649ff79832566293cb5cea9da"
)
STATIONARITY_PATH = (
    ROOT / "results/isj_v8/dev_0027_stationarity_audit_v1"
    / "stationarity_results.json"
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_preflight() -> tuple[dict, dict[str, np.ndarray]]:
    """No graph solve, reference read or output folder creation."""
    if OUT.exists():
        raise FileExistsError(f"V8.3.3 output exists; refuse overwrite: {OUT}")
    frozen, summary = load_sealed_source_report()
    if not summary["all_stable"]:
        raise ValueError("source V8.3 results not fully auditable")
    if (not STATIONARITY_PATH.is_file()
            or file_sha256(STATIONARITY_PATH) !=
            EXPECTED_V8_3_STATIONARITY_SHA):
        raise ValueError("source V8.3.2 stationarity JSON SHA256 mismatch")
    old_gradient_data = json.loads(
        STATIONARITY_PATH.read_text(encoding="utf-8"))
    if (old_gradient_data.get("source_v8_3_1_sha256")
            != EXPECTED_PREVIOUS_SHA256
            or old_gradient_data.get("heldout_0016_0020_accessed") is not False):
        raise ValueError("V8.3.2 provenance mismatch")
    prepared, sigmas = prepare_cases()
    if (prepared.get("all_cases_use_same_frame_range") != [3180, 3860]
            or prepared.get("heldout_0016_0020_accessed") is not False
            or tuple(sigmas) != CASES):
        raise ValueError("V8.3.3 exact DEV case set mismatch")
    for key in (
        "gps_degraded_sha256", "gps_candidate_sha256",
        "lidar_factors_sha256", "camera_factors_sha256",
    ):
        if frozen.get(key) != prepared.get(key):
            raise ValueError(f"V8.3.3 source file hash mismatch: {key}")
    for case in CASES:
        sigma = np.asarray(sigmas[case], np.float64)
        frozen_stats = frozen["case_sigmas"][case]
        if (sigma.shape != (680, 3) or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)
                or not math.isclose(
                    float(sigma.min()), frozen_stats["min_m"],
                    rel_tol=0, abs_tol=1e-8)
                or not math.isclose(
                    float(sigma.max()), frozen_stats["max_m"],
                    rel_tol=0, abs_tol=1e-8)
                or not np.allclose(
                    sigma.mean(axis=0), frozen_stats["mean_xyz_m"],
                    rtol=0, atol=1e-8)):
            raise ValueError(f"{case}: frozen 0027 GPS covariance changed")
    metadata = {
        "schema": "v8_0027_dev_solver_health_v1",
        "status": "PREFLIGHT_ONLY",
        "development_only": True,
        "posthoc_numeric_replay": True,
        "heldout_0016_0020_read": False,
        "independent_position_gt": False,
        "offline_batch_lm_not_online_isam2": True,
        "sensor_factors_unchanged": True,
        "gps_cap4_unchanged": True,
        "old_V7_V8_results_immutable": True,
        "source_v8_3_sha256": file_sha256(
            ROOT / "results/isj_v8/dev_0027_feedforward_fgo_v1"
            / "v8_dev_feedforward_fgo_results.json"),
        "source_v8_3_2_sha256": EXPECTED_V8_3_STATIONARITY_SHA,
        "cases": list(CASES),
        "initialization_names": list(SEED_NAMES),
        "initialization_selection_uses_OXTS": False,
        "covariance_selection_uses_OXTS": False,
        "candidate_selection": "min_same_target_objective_only_when_CONVERGED_CANDIDATE",
        "all_results_are_development_only": True,
        "new_pose_archive": str(POSE_ARCHIVE),
    }
    return metadata, sigmas


def common_initializations(gtsam, original, degraded_gps: np.ndarray):
    """Return exact same two initializers for all 16 objectives."""
    gps = np.asarray(degraded_gps, dtype=np.float64)
    if (gps.shape != (680, 3) or not np.isfinite(gps).all()
            or np.linalg.norm(gps[0]) > 1e-8):
        raise ValueError("expected 0027 subgraph degraded GPS rebased to origin")
    poses = _positions(gtsam, original, 680)
    if poses.shape != (680, 4, 4):
        raise ValueError("unexpected LiDAR initializer geometry")
    hybrid = poses.copy()
    hybrid[:, :3, 3] = gps
    # GPS(t) is a measured, degraded observation. Not clean OXTS truth.
    gps_seed = build_values_from_pose_matrices(gtsam, hybrid)
    return {
        SEED_NAMES[0]: original,
        SEED_NAMES[1]: gps_seed,
    }


def run_replay() -> dict:
    metadata, sigmas = source_preflight()
    import gtsam

    measurement = load_degraded_four_sensor_measurements(DEGRADED)
    sub = subgraph_measurements(measurement, begin=BEGIN, end=END)
    reference_independent_seeds = None
    original_pose = None
    record = {}
    pose_arrays = {}
    selected_poses = {}

    # Execute ALL solves before opening the OXTS/GPS-correlated reference.
    for case, sigma in sigmas.items():
        graph, original, counts = run_four_sensor_factor_graph(
            sub, OUT / "NO_GRAPH_WRITES", mode="fixed",
            gps_sigmas_override=sigma, build_graph_only=True,
        )
        if counts != FACTORS:
            raise ValueError(f"{case}: sensor factor counts drifted")
        base_poses = _positions(gtsam, original, 680)
        if original_pose is None:
            original_pose = base_poses.copy()
            reference_independent_seeds = common_initializations(
                gtsam, original, sub.gps_local)
            pose_arrays["seed_lidar"] = original_pose.copy()
            pose_arrays["seed_degraded_gps"] = _positions(
                gtsam, reference_independent_seeds[SEED_NAMES[1]], 680)
        elif not np.array_equal(original_pose, base_poses):
            raise ValueError(f"{case}: base LiDAR initialization changed")
        observations = {}
        # We do not optimize after reading reference.
        for seed_name in SEED_NAMES:
            init = reference_independent_seeds[seed_name]
            try:
                solved = solve_v8_health(graph, init, 680)
                observations[seed_name] = {
                    "status": solved["status"],
                    "objective": solved["objective"],
                    "initial_objective": solved["initial_objective"],
                    "initial_gradient_l2": solved["initial_gradient_l2"],
                    "terminal_gradient_l2": solved["final_gradient_l2"],
                    "passes": solved["passes"],
                    "candidate_not_stationarity_certificate": True,
                }
                pose_arrays[f"{case}__{seed_name}"] = solved["poses"].copy()
            except Exception as exc:
                # Preserve a failed candidate without upgrading it into a
                # valid solution. A separate case may still succeed.
                observations[seed_name] = {
                    "status": "FAILED_NUMERICAL",
                    "exception_class": type(exc).__name__,
                    "exception_summary": str(exc)[:250],
                    "candidate_not_stationarity_certificate": True,
                }
            print(
                "V8_3_3_SOLVER", case, seed_name,
                observations[seed_name]["status"],
                observations[seed_name].get("objective", None), flush=True)
        decision = select_reliable_candidate(observations)
        if decision["selected_seed"] is not None:
            selected_poses[case] = pose_arrays[
                f"{case}__{decision['selected_seed']}"]
        record[case] = {
            "initializers": observations,
            "selection": decision,
        }

    # No solver decision may depend on reference; only retrospective
    # OXTS/GPS-correlated development diagnostics from this point onward.
    with np.load(SOURCE, allow_pickle=False) as z:
        ref = np.asarray(z["clean_gps_local"][BEGIN:END], np.float64)
    if ref.shape != (680, 3) or not np.isfinite(ref).all():
        raise ValueError("unexpected reference evaluation dimensions")
    for case in CASES:
        poses = selected_poses.get(case)
        record[case]["retrospective_oxTS_correlated_evaluation"] = (
            reference_diagnostics(poses[:, :3, 3], ref)
            if poses is not None else None
        )
    metadata["case_results"] = record
    metadata["candidate_case_count"] = len(selected_poses)
    metadata["cases_without_qualified_candidate"] = [
        case for case in CASES if case not in selected_poses
    ]
    metadata["evaluation_not_independent"] = True
    metadata["solver_health_caveat"] = (
        "CONVERGED_CANDIDATE is an engineering stop, "
        "not a mathematical stationarity certificate."
    )
    metadata["status"] = "COMPLETE_0027_DEV_NUMERICAL_HEALTH_REPLAY"

    # Write once in a new folder; old experiment/result files untouched.
    OUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(POSE_ARCHIVE, **pose_arrays)
    metadata["pose_archive_sha256"] = file_sha256(POSE_ARCHIVE)
    METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": metadata["status"],
        "qualified_cases": metadata["candidate_case_count"],
        "total_cases": len(CASES),
        "not_qualified": metadata["cases_without_qualified_candidate"],
        "source_V8_3_sha256": metadata["source_v8_3_sha256"],
        "new_result_sha256": file_sha256(METADATA),
        "poses_sha256": metadata["pose_archive_sha256"],
    }, indent=2, ensure_ascii=False), flush=True)
    print(
        "V8_DEV_SOLVER_HEALTH_REPLAY_COMPLETE; 16_CASES; 32_SOLVES; "
        "0027_DEV_ONLY; NO_HELDOUT; NOT_INDEPENDENT",
        flush=True,
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-replay", action="store_true")
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)
    if args.preflight:
        if args.token is not None:
            parser.error("no token for read-only preflight")
        data, _ = source_preflight()
        print(json.dumps(data, indent=2, ensure_ascii=False), flush=True)
        print("V8_3_3_PREFLIGHT_OK; NO_SOLVES; NO_WRITES; NO_HELDOUT",
              flush=True)
        return
    if args.token != TOKEN:
        parser.error("explicit 0027 DEV-only solver replay token required")
    run_replay()


if __name__ == "__main__":
    main()
