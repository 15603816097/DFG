from __future__ import annotations

"""Read-only, equal-budget multi-initialization V7 FGO DEV comparison.

Source: KITTI 0027 only, GPS h0 anchors [3180,3860). The identical
680-frame GPS/IMU/LiDAR/Camera factor measurements are used in every
method. It is an OFFLINE BATCH FGO comparison on an already explored
development validation interval, *not* independent truth or iSAM2.

Each method is optimized from the SAME, frozen pool of initial states:
(1) LiDAR-integrated initial poses, (2) saved fixed-GPS poses and
(3) saved Mamba-R poses from seed 43. The latter two were explored
earlier, hence this entire protocol is explicitly POST-HOC.

For each *method's own objective*, select the converged or best-observed
lowest-objective candidate WITHOUT using clean OXTS, synthetic oracle
labels, or any reference metric in optimization or selection.
Only after ALL methods are solved is the measurement-correlated clean
OXTS reference loaded for descriptive diagnostics.

No checkpoint, data file, or old FGO output is ever written or modified.
Use --run only after the read-only source preflight succeeds.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, CASES, DEGRADED, END, OUTPUT, prepare_case,
    reference_diagnostics, subgraph_measurements,
)
from experiments.isj_v7.compare_dev_gps import sha256
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import (
    _pose3_from_matrix, _pose3_to_matrix, run_four_sensor_factor_graph,
)

N = END - BEGIN


# Data-integrity and pose helpers formerly kept in the one-off
# audit_fg_convergence.py audit. Uniform multi-start is now their only
# active caller; colocating them removes that superseded script.
def verify_pose_array(poses: np.ndarray) -> np.ndarray:
    a = np.asarray(poses, dtype=np.float64)
    if a.shape != (N, 4, 4) or not np.isfinite(a).all():
        raise ValueError("invalid saved FGO pose array")
    if not np.allclose(a[:, 3, :], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("nonhomogeneous saved FGO poses")
    rots = a[:, :3, :3]
    if (np.max(np.abs(rots @ rots.transpose(0, 2, 1)
                      - np.eye(3))) > 1e-5
            or np.max(np.abs(np.linalg.det(rots) - 1.0)) > 1e-5):
        raise ValueError("invalid saved FGO rotations")
    return a


def values_from_poses(gtsam, poses: np.ndarray):
    verified = verify_pose_array(poses)
    values = gtsam.Values()
    for i, p in enumerate(verified):
        values.insert(i, _pose3_from_matrix(gtsam, p))
    return values


def load_saved_case(case: str, seed: int, expected_sigma: np.ndarray,
                    expected_meta: dict) -> tuple[np.ndarray, dict]:
    folder = OUTPUT / f"seed_{seed}" / case
    p = folder / "poses.npy"
    metric_path = folder / "pilot_metrics.json"
    traj_path = folder / "trajectory.txt"
    if not (p.is_file() and metric_path.is_file() and traj_path.is_file()):
        raise FileNotFoundError(f"incomplete saved pilot: {folder}")
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    if (metrics.get("schema") != "isj_v7_0027_dev_gps_fgo_pilot_v1"
            or metrics.get("status") != "COMPLETE_DEVELOPMENT_PILOT"
            or metrics.get("development_only") is not True
            or metrics.get("reference_is_not_independent") is not True
            or metrics.get("interval_half_open") != [BEGIN, END]
            or metrics.get("frames") != N
            or metrics.get("case") != case
            or metrics.get("seed") != seed
            or metrics.get("gps_measurement_archive_sha256")
                 != expected_meta["gps_measurement_archive_sha256"]
            or metrics.get("dataset_archive_sha256")
                 != expected_meta["dataset_archive_sha256"]
            or metrics.get("prediction_archive_sha256")
                 != expected_meta["prediction_archive_sha256"]
            or metrics.get("trajectory_sha256") != sha256(traj_path)):
        raise ValueError(f"invalid saved provenance: {folder}")
    poses = verify_pose_array(np.load(p, allow_pickle=False))
    stored_trajectory = np.loadtxt(traj_path, dtype=np.float64)
    if (stored_trajectory.shape != (N, 3)
            or not np.allclose(poses[:, :3, 3], stored_trajectory,
                               atol=2e-7, rtol=0)):
        raise ValueError("saved pose and trajectory disagree")
    sigma_file = folder / "gps_sigmas.npy"
    if case == "no_gps":
        if sigma_file.exists():
            raise ValueError("no_gps should not store GPS sigmas")
    else:
        if not sigma_file.is_file():
            raise FileNotFoundError(f"missing GPS sigma archive: {folder}")
        sigmas = np.load(sigma_file, allow_pickle=False)
        if not np.array_equal(sigmas, expected_sigma):
            raise ValueError("stored GPS sigmas changed after pilot")
    return poses, metrics


START_NAMES = ("lidar", "fixed_saved", "mamba_saved")
DEFAULT_CASES = (
    "fixed", "gru96_r", "gru106_r", "mamba_r",
    "mamba_r_u", "mamba_mean_sigma", "mamba_perm_sigma",
)
DEFAULT_MAX_PASSES = 4
REL_OBJ_TOL = 1e-6
POSITION_SHIFT_TOL_M = .01


def check_protocol(seed: int, cases: list[str], passes: int,
                   starts: tuple[str, ...]) -> None:
    if seed != 43:
        raise ValueError(
            "Uniform V7 audit uses seed43 original common starts only")
    if (len(cases) == 0 or len(set(cases)) != len(cases)
            or any(c not in CASES for c in cases)):
        raise ValueError("invalid or duplicate V7 method cases")
    if passes < 2 or passes > 12:
        raise ValueError("passes must be between 2 and 12")
    if starts != START_NAMES:
        raise ValueError(
            "all three predeclared common initializations are required")


def _pose_from_values(gtsam, values) -> np.ndarray:
    p = np.stack(
        [_pose3_to_matrix(values.atPose3(i)) for i in range(N)], axis=0)
    return verify_pose_array(p)


def solve_from_start(graph, initial, *, passes: int,
                     rel_tol: float = REL_OBJ_TOL,
                     shift_tol_m: float = POSITION_SHIFT_TOL_M) -> dict:
    """Same 100-iteration GTSAM LM budget and tolerances for every pass.

    Do not choose starts using OXTS. Stop only after objective decrease
    AND positional shift are both small. 'stabilized' is numeric only,
    not proof of global optimum or unique solution.
    """
    import gtsam
    if (passes < 2 or passes > 12
            or not (0 < rel_tol < 1)
            or not np.isfinite(shift_tol_m)
            or shift_tol_m <= 0):
        raise ValueError("invalid solver convergence protocol")
    values = initial
    history = []
    stabilized = False
    began = time.monotonic()
    previous_pose = _pose_from_values(gtsam, values)
    for idx in range(passes):
        before = float(graph.error(values))
        cfg = gtsam.LevenbergMarquardtParams()
        cfg.setMaxIterations(100)
        cfg.setRelativeErrorTol(1e-7)
        optimized = gtsam.LevenbergMarquardtOptimizer(
            graph, values, cfg).optimize()
        after = float(graph.error(optimized))
        current_pose = _pose_from_values(gtsam, optimized)
        if not np.isfinite([before, after]).all():
            raise RuntimeError("nonfinite factor-graph objective")
        allowed_numerical_upturn = 1e-8 * max(abs(before), 1.0)
        if after > before + allowed_numerical_upturn:
            raise RuntimeError(
                f"LM objective unexpectedly increased: {before} -> {after}")
        pos_diff = (current_pose[:, :3, 3]
                    - previous_pose[:, :3, 3])
        shift_rmse = float(np.sqrt(
            np.mean(np.sum(pos_diff * pos_diff, axis=1))))
        change_ratio = max(0.0, (before - after)) / max(
            abs(before), 1.0)
        stable_this_pass = (
            change_ratio <= rel_tol and shift_rmse <= shift_tol_m)
        history.append({
            "pass": idx + 1,
            "objective_before": before,
            "objective_after": after,
            "relative_objective_reduction": float(change_ratio),
            "position_shift_rmse_m": shift_rmse,
            "stabilized": bool(stable_this_pass),
        })
        values = optimized
        previous_pose = current_pose
        if stable_this_pass:
            stabilized = True
            break
    return {
        "objective": float(graph.error(values)),
        "stabilized": stabilized,
        "passes": history,
        "elapsed_seconds": float(time.monotonic() - began),
        "poses": previous_pose,
    }


def select_within_graph_objective(
        candidates: dict[str, dict]) -> tuple[str, dict]:
    """Only each graph's objective; never reference RMSE."""
    if tuple(candidates) != START_NAMES:
        raise ValueError("candidate pool must be the common start pool")
    for name, row in candidates.items():
        value = row.get("objective")
        if value is None or not math.isfinite(value):
            raise ValueError(f"invalid candidate objective {name}")
    chosen = min(START_NAMES,
                 key=lambda name: float(candidates[name]["objective"]))
    return chosen, candidates[chosen]


def load_common_starts(seed: int) -> dict[str, np.ndarray]:
    result = {}
    for case, label in (("fixed", "fixed_saved"),
                        ("mamba_r", "mamba_saved")):
        sigma, meta = prepare_case(case, seed)
        poses, _ = load_saved_case(case, seed, sigma, meta)
        result[label] = poses.copy()
    return result


def analyze(cases: list[str], *, seed: int,
            max_passes: int, run_solver: bool) -> dict:
    check_protocol(seed, cases, max_passes, START_NAMES)
    # Source preflight all methods BEFORE running any optimizer.
    # It does not open clean OXTS.
    source_checked = {
        case: prepare_case(case, seed) for case in cases
    }
    common = load_common_starts(seed)
    if not run_solver:
        for case in cases:
            sigma, meta = source_checked[case]
            print(
                f"PREFLIGHT_OK case={case} frames={N} "
                f"GPS_factors={case != 'no_gps'} "
                f"sigma_mean={meta.get('sigma_mean')}",
                flush=True,
            )
        return {
            "development_only": True,
            "preflight_only": True,
            "source_verified": True,
            "cases": cases,
            "start_names": list(START_NAMES),
            "seed": seed,
            "writes": False,
        }
    m = load_degraded_four_sensor_measurements(DEGRADED)
    segment = subgraph_measurements(m)
    results = {}
    finished = {}
    for case in cases:
        sigma, meta = source_checked[case]
        graph, initial, factor_counts = run_four_sensor_factor_graph(
            segment, OUTPUT / "__UNIFORM_AUDIT_NO_WRITE__",
            mode="fixed", use_gps=(case != "no_gps"),
            gps_sigmas_override=(None if case == "no_gps" else sigma),
            use_imu=True, use_lidar=True, use_camera=True,
            build_graph_only=True,
        )
        import gtsam
        init_poses = _pose_from_values(gtsam, initial)
        start_pool = {
            "lidar": init_poses,
            "fixed_saved": common["fixed_saved"],
            "mamba_saved": common["mamba_saved"],
        }
        candidates = {}
        for start_name in START_NAMES:
            initial_values = values_from_poses(gtsam, start_pool[start_name])
            sol = solve_from_start(graph, initial_values, passes=max_passes)
            candidates[start_name] = sol
            print(
                f"SOLVE case={case} start={start_name} "
                f"objective={sol['objective']:.6f} "
                f"passes={len(sol['passes'])} "
                f"stabilized={sol['stabilized']}",
                flush=True,
            )
        selected_name, selected = select_within_graph_objective(
            candidates)
        result = {
            "case": case, "seed": seed,
            "schema": "isj_v7_0027_uniform_multistart_fgo_audit_v1",
            "development_only": True,
            "independent_test": False,
            "online_incremental_fgo": False,
            "time_range_half_open": [BEGIN, END],
            "start_pool": list(START_NAMES),
            "post_hoc_common_start_pool": True,
            "selected_by": "min graph objective within each case only",
            "selected_start": selected_name,
            "selected_objective": selected["objective"],
            "selected_stabilized": selected["stabilized"],
            "all_starts_stabilized": all(
                c["stabilized"] for c in candidates.values()),
            "factor_counts": factor_counts,
            "source_measurement_sha256": meta[
                "gps_measurement_archive_sha256"],
            "prediction_archive_sha256": meta[
                "prediction_archive_sha256"],
            "per_start": {
                name: {
                    "objective": x["objective"],
                    "stabilized": x["stabilized"],
                    "passes": x["passes"],
                    "elapsed_seconds": x["elapsed_seconds"],
                } for name, x in candidates.items()
            },
            "evaluation": None,
        }
        results[case] = result
        finished[case] = selected["poses"]
    # All solver runs/within-objective selections are finished.
    # Load nonindependent reference ONLY now, for descriptive scoring.
    with np.load(
        DEGRADED / "degraded_sensor_data.npz", allow_pickle=False,
    ) as data:
        clean = np.asarray(
            data["clean_gps_local"][BEGIN:END], dtype=np.float64).copy()
    for case in cases:
        res = results[case]
        res["evaluation"] = reference_diagnostics(
            finished[case][:, :3, 3], clean)
        print(
            f"RESULT case={case} best_start={res['selected_start']} "
            f"objective={res['selected_objective']:.6f} "
            f"stable={res['selected_stabilized']} "
            f"dev_oxts_3d_rmse={res['evaluation']['origin_aligned_oxTS_relative_3d_rmse_m']:.6f}",
            flush=True,
        )
    return {
        "schema": "isj_v7_uniform_multistart_audit_v1",
        "development_only": True,
        "reused_explored_0027_val": True,
        "reference_is_independent": False,
        "reference_used_for_selection": False,
        "read_only_no_writes": True,
        "seed": seed,
        "max_passes_per_start": max_passes,
        "LM_max_iters_per_pass": 100,
        "LM_rel_error_tol": 1e-7,
        "numeric_objective_relative_stable_tol": REL_OBJ_TOL,
        "numeric_position_shift_stable_tol_m": POSITION_SHIFT_TOL_M,
        "models": results,
        "conclusion_guard": (
            "Cannot claim superior state estimation if solver not stabilized, "
            "and OXTS reference is not independent of GPS."),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    p.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    p.add_argument("--run", action="store_true",
                   help="solve factor graphs in memory; never write project files")
    args = p.parse_args(argv)
    report = analyze(
        args.cases, seed=args.seed, max_passes=args.max_passes,
        run_solver=args.run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("V7_UNIFORM_MULTI_START_"
          + ("COMPLETE" if args.run else "PREFLIGHT_COMPLETE")
          + "; DEV_ONLY; NO_FGO_FILES_WRITTEN; NO_HELDOUT", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
