from __future__ import annotations

"""Read-only V7 0027 FGO convergence diagnosis on saved 680-frame runs.

Uses *exactly* the same factor graph builder, data and per-frame GPS
sigmas as the initial V7 dev pilot. Evaluates the saved pose estimate
under its OWN objective and optionally continues LM from that saved
estimate or from a different saved dev-pilot solution.

The 680-frame archive is already explored validation; OXTS is not
independent reference and is NEVER an optimization input. No files
are written by this command. Never accesses 0016 or 0020.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.run_gps_covariance_pilot import (
    BEGIN, CASES, DEGRADED, END, OUTPUT, prepare_case,
    subgraph_measurements,
)
from experiments.isj_v7.compare_dev_gps import sha256
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import (
    _pose3_from_matrix, _pose3_to_matrix,
    run_four_sensor_factor_graph,
)

N = END - BEGIN


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


def _continue_from(graph, gtsam, values, original_poses: np.ndarray) -> dict:
    cfg = gtsam.LevenbergMarquardtParams()
    cfg.setMaxIterations(100)
    cfg.setRelativeErrorTol(1e-7)
    before = float(graph.error(values))
    refined = gtsam.LevenbergMarquardtOptimizer(
        graph, values, cfg).optimize()
    after = float(graph.error(refined))
    refined_poses = np.stack(
        [_pose3_to_matrix(refined.atPose3(i)) for i in range(N)], axis=0)
    if not np.isfinite([before, after]).all():
        raise RuntimeError("nonfinite graph error in continuation")
    shift = (refined_poses[:, :3, 3]
             - verify_pose_array(original_poses)[:, :3, 3])
    return {
        "objective_before": before,
        "objective_after": after,
        "objective_improvement": before - after,
        "position_shift_3d_rmse_m": float(np.sqrt(
            np.mean(np.sum(shift * shift, axis=1)))),
        "position_shift_3d_max_m": float(np.max(
            np.linalg.norm(shift, axis=1))),
    }


def audit_case(case: str, seed: int, *, refine: bool,
               cross_start: str | None = None) -> dict:
    if case not in CASES or seed not in (43, 44, 45):
        raise ValueError("invalid V7 dev case or seed")
    if cross_start is not None and cross_start not in CASES:
        raise ValueError("invalid cross-start model case")
    sigma, expected_meta = prepare_case(case, seed)
    stored_poses, metadata = load_saved_case(
        case, seed, sigma, expected_meta)
    m = load_degraded_four_sensor_measurements(DEGRADED)
    sub = subgraph_measurements(m)
    graph, original_init, factors = run_four_sensor_factor_graph(
        sub, OUTPUT / "__READ_ONLY_NEVER_CREATE__", mode="fixed",
        use_gps=(case != "no_gps"), use_imu=True,
        use_lidar=True, use_camera=True,
        gps_sigmas_override=(None if case == "no_gps" else sigma),
        build_graph_only=True,
    )
    if {k: int(v) for k, v in factors.items()} != metadata["factor_counts"]:
        raise ValueError("factor inventory changed from saved V7 pilot")
    import gtsam
    saved_values = values_from_poses(gtsam, stored_poses)
    init_obj = float(graph.error(original_init))
    saved_obj = float(graph.error(saved_values))
    if not np.isfinite([init_obj, saved_obj]).all():
        raise RuntimeError("nonfinite factor graph objective")
    result = {
        "case": case,
        "seed": seed,
        "development_only": True,
        "reference_is_not_independent": True,
        "read_only": True,
        "frames": N,
        "graph_factors": int(graph.size()),
        "objective_from_original_lidar_initial": init_obj,
        "objective_from_saved_solution": saved_obj,
        "saved_3d_rmse_to_non_independent_oxts_m": metadata[
            "evaluation"]["origin_aligned_oxTS_relative_3d_rmse_m"],
        "continued_from_saved": None,
        "continued_from_alternative_start": None,
        "warning": (
            "Per-case objective scales differ due to different GPS noise; "
            "only compare before/after under the SAME graph."
        ),
    }
    if refine:
        result["continued_from_saved"] = _continue_from(
            graph, gtsam, saved_values, stored_poses)
    if cross_start is not None and cross_start != case:
        alt_sigma, alt_meta = prepare_case(cross_start, seed)
        alt_poses, _ = load_saved_case(
            cross_start, seed, alt_sigma, alt_meta)
        alt_values = values_from_poses(gtsam, alt_poses)
        cross_result = _continue_from(
            graph, gtsam, alt_values, stored_poses)
        cross_result["alternative_start_case"] = cross_start
        result["continued_from_alternative_start"] = cross_result
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", choices=CASES,
                   default=["fixed", "mamba_r",
                            "mamba_mean_sigma", "mamba_perm_sigma"])
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--refine", action="store_true",
                   help="run LM continuation in memory only, no writes")
    p.add_argument("--cross-start", choices=CASES,
                   help="restart this case's objective from a saved different case")
    args = p.parse_args(argv)
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("duplicate cases")
    for name in args.cases:
        print(json.dumps(
            audit_case(name, args.seed, refine=args.refine,
                       cross_start=args.cross_start),
            indent=2, ensure_ascii=False), flush=True)
    print("V7_FGO_OBJECTIVE_AUDIT_COMPLETE; READ_ONLY; NO_WRITES; NO_HELDOUT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
