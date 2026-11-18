from __future__ import annotations

"""Frozen Huber/Cauchy Stage-7 baselines on isolated cross-date KITTI drives.

Copies the original Stage-7 robust GPS protocol: same physical factors,
fixed sigmas and stage-one Fixed-FGO initialization; k=1.345 (Huber) and
k=2.3849 (Cauchy). Adds optional IMU/Camera robust kernels; LiDAR remains
fixed Gaussian. This is OFFLINE two-stage FGO, not streaming.

No checkpoint training, GT initialization, model parameter tuning, or
overwriting previous outputs. Drive 0016 is already inspected EXPLORATORY.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence import run_stage7_factor_graph as s7
from experiments.cross_sequence.run_stage7_robust_baselines import KERNELS as ORIGINAL_KERNELS
from experiments.cross_sequence.stage7_config import (
    GPS_FIXED, IMU_R0, LIDAR_R0, LIDAR_T0, CAM_R0, CAM_T0,
)
from experiments.isj_supplement.run_cross_date import (
    RESULT_ROOT, crossdate_bindings, date_for, require_processed,
    result_is_complete, validate_destination,
)
from experiments.isj_supplement.run_cross_date_beta_swap import (
    write_new_summary, save_protocol,
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig

KERNELS = dict(ORIGINAL_KERNELS)
SCOPES = {"gps": ("gps",), "all": ("gps", "imu", "camera")}
RTOL = 1e-7
SHIFT_TOL = 1e-3
MAX_ITER = 100


def specs_for(kernels: str, scope: str) -> list[dict]:
    names = list(KERNELS) if kernels == "both" else [kernels]
    modes = list(SCOPES) if scope == "both" else [scope]
    if any(x not in KERNELS for x in names) or any(x not in SCOPES for x in modes):
        raise ValueError("Unknown robust kernel or scope")
    return [
        {
            "name": f"ISJ_ROBUST_{kernel.upper()}_{mode.upper()}",
            "kernel": kernel,
            "k": float(KERNELS[kernel]),
            "scope": mode,
            "robust_sensors": list(SCOPES[mode]),
        }
        for kernel in names for mode in modes
    ]


def make_graph(m, fixed: np.ndarray, spec: dict):
    """Preserve every Stage-7 factor, applying Robust noise only to requested sensors."""
    import gtsam

    n = len(m.gps_local)
    if fixed.shape != (n, 4, 4) or not np.isfinite(fixed).all():
        raise ValueError("Fixed trajectory shape or values invalid")
    cfg = FourSensorFactorConfig()
    selected = set(spec["robust_sensors"])
    if not selected.issubset({"gps", "imu", "camera"}):
        raise ValueError("Unrecognized robust sensor")

    cls = getattr(gtsam.noiseModel.mEstimator, spec["kernel"])
    kernel = cls.Create(float(spec["k"]))

    def noise(sigmas, sensor):
        base = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(sigmas, dtype=np.float64)
        )
        if sensor in selected:
            return gtsam.noiseModel.Robust.Create(kernel, base)
        return base

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    anchor = s7.initial_pose_from_oxts(gtsam, m)
    graph.add(gtsam.PriorFactorPose3(
        0, anchor,
        gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray([cfg.prior_rotation_sigma]*3 +
                       [cfg.prior_translation_sigma]*3, dtype=float)
        )
    ))
    for i in range(n):
        values.insert(i, s7._pose3_from_matrix(gtsam, fixed[i]))

    gps_noise = noise([1e6]*3 + [GPS_FIXED]*3, "gps")
    imu_noise = noise([IMU_R0]*3 + [cfg.imu_translation_sigma]*3, "imu")
    cam_noise = noise([CAM_R0]*3 + [CAM_T0]*3, "camera")
    lidar_noise = noise([LIDAR_R0]*3 + [LIDAR_T0]*3, "lidar")

    counts = {"anchor": 1, "gps": 0, "imu": 0, "camera": 0, "lidar": 0}
    ts = np.asarray(m.timestamps, dtype=float)
    for i in range(n):
        gps_pose = gtsam.Pose3(
            gtsam.Rot3(),
            gtsam.Point3(*map(float, m.gps_local[i, :3]))
        )
        graph.add(gtsam.PriorFactorPose3(i, gps_pose, gps_noise))
        counts["gps"] += 1
        if i == n - 1:
            continue

        dt = float(ts[i+1] - ts[i])
        if not np.isfinite(dt) or dt <= 0 or dt > 1:
            dt = 0.1
        graph.add(gtsam.BetweenFactorPose3(
            i, i+1, s7._imu_between_pose(gtsam, m.imu_gyro[i], dt),
            imu_noise
        ))
        counts["imu"] += 1
        if bool(m.lidar_valid[i]):
            graph.add(gtsam.BetweenFactorPose3(
                i, i+1, s7._pose3_from_matrix(gtsam, m.lidar_between[i]),
                lidar_noise
            ))
            counts["lidar"] += 1
        if bool(m.camera_valid[i]):
            graph.add(gtsam.BetweenFactorPose3(
                i, i+1, s7._pose3_from_matrix(gtsam, m.camera_between[i]),
                cam_noise
            ))
            counts["camera"] += 1
    if graph.size() != sum(counts.values()):
        raise AssertionError("Wrong robust factor topology")
    return graph, values, counts


def check_cache(out: Path, sequence: str, spec: dict, n: int, *,
                stage: str) -> dict | None:
    folder = out / "factor_graph" / (
        spec["name"] if stage == "fgo"
        else "ISJ_CONVERGENCE_" + spec["name"]
    )
    if not folder.exists():
        return None
    pose_path, meta_path = folder / "poses.npy", folder / "metrics.json"
    if not pose_path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"Partial cached result; refuse overwrite: {folder}")
    poses = np.load(pose_path, allow_pickle=False)
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"Corrupt cached robust poses: {pose_path}")
    info = json.loads(meta_path.read_text(encoding="utf-8"))
    if info.get("sequence") != sequence or info.get("spec") != spec:
        raise ValueError(f"Cached robust method provenance mismatch: {folder}")
    if info.get("stage") != stage:
        raise ValueError(f"Cached robust stage mismatch: {folder}")
    if stage == "converge":
        required = ("final_metrics", "stationary_under_rule", "max_extra_passes")
    else:
        required = ("metrics", "graph_final_objective", "graph_initial_objective")
    for key in required:
        if key not in info:
            raise ValueError(f"Missing {key} from cached result {meta_path}")
    return info


def solve(gtsam, graph, values):
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(MAX_ITER)
    p.setRelativeErrorTol(RTOL)
    start = time.perf_counter()
    solved = gtsam.LevenbergMarquardtOptimizer(graph, values, p).optimize()
    return solved, time.perf_counter() - start


def poses_from_values(values, n: int):
    return np.stack(
        [s7._pose3_to_matrix(values.atPose3(i)) for i in range(n)]
    )


def initial_data(sequence: str, out: Path):
    n = require_processed(out)
    if not result_is_complete(out, "FINAL_S0_fixed", n):
        raise FileNotFoundError("Run fixed FGO before robust baseline")
    saved = np.load(
        out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy",
        allow_pickle=False,
    )
    measurements = s7.load_measurements(sequence)
    if len(measurements.gps_local) != n:
        raise ValueError("Sequence length mismatch")
    return measurements, saved, n


def write_new_case(out: Path, spec: dict, poses: np.ndarray, meta: dict,
                   *, stage: str) -> None:
    folder = out / "factor_graph" / (
        spec["name"] if stage == "fgo"
        else "ISJ_CONVERGENCE_" + spec["name"]
    )
    folder.mkdir(parents=True, exist_ok=False)
    np.save(folder / "poses.npy", poses)
    np.savetxt(folder / "trajectory.txt", poses[:, :3, 3], fmt="%.9f")
    (folder / "metrics.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_fgo(sequence: str, out: Path, specs: list[dict]):
    import gtsam
    m, fixed, n = initial_data(sequence, out)
    rows = []
    for spec in specs:
        info = check_cache(out, sequence, spec, n, stage="fgo")
        if info is None:
            graph, values, counts = make_graph(m, fixed, spec)
            initial_objective = float(graph.error(values))
            solved, seconds = solve(gtsam, graph, values)
            final_objective = float(graph.error(solved))
            if not all(map(math.isfinite, (initial_objective, final_objective))):
                raise ValueError("Nonfinite robust objective")
            if final_objective > initial_objective + max(1e-7, abs(initial_objective)*1e-10):
                raise RuntimeError("Robust optimizer increased objective")
            poses = poses_from_values(solved, n)
            metrics = s7.evaluate(sequence, poses[:, :3, 3])
            info = {
                "sequence": sequence,
                "spec": spec,
                "stage": "fgo",
                "protocol": "frozen_crossdate_robust_m_estimator",
                "first_stage": "FINAL_S0_fixed",
                "ground_truth_used_for_weights_or_initialization": False,
                "gps_pseudo_prior_rotation_sigma": 1e6,
                "lidar_robustified": False,
                "factor_counts": counts,
                "graph_initial_objective": initial_objective,
                "graph_final_objective": final_objective,
                "solver_seconds": seconds,
                "metrics": metrics,
                "classification": "exploratory_post_hoc_not_blind",
            }
            write_new_case(out, spec, poses, info, stage="fgo")
            print(f"{spec['name']}: new", flush=True)
        else:
            print(f"{spec['name']}: cached", flush=True)
        ate = float(info["metrics"]["ATE3D"])
        rows.append({"sequence": sequence, "case": spec["name"], "ATE3D_m": ate})
        print(f"  {spec['name']:<34} ATE3D={ate:.6f}m", flush=True)

    label = "_".join(s["name"] for s in specs)
    write_new_summary(out / f"isj_cross_date_robust_fgo_{label}.csv", rows)
    return rows


def run_converge(sequence: str, out: Path, specs: list[dict], max_passes: int):
    import gtsam
    m, fixed, n = initial_data(sequence, out)
    rows = []
    for spec in specs:
        raw = check_cache(out, sequence, spec, n, stage="fgo")
        if raw is None:
            raise FileNotFoundError(f"Run --stage fgo first: {spec['name']}")
        saved = check_cache(out, sequence, spec, n, stage="converge")
        if saved is not None:
            if int(saved["max_extra_passes"]) != max_passes:
                raise ValueError("Existing convergence budget differs; no overwrite")
            print(f"{spec['name']}: converged result cached", flush=True)
        else:
            graph, _, counts = make_graph(m, fixed, spec)
            poses = np.load(
                out / "factor_graph" / spec["name"] / "poses.npy",
                allow_pickle=False,
            )
            values = gtsam.Values()
            for i, T in enumerate(poses):
                values.insert(i, s7._pose3_from_matrix(gtsam, T))
            old_cost = float(graph.error(values))
            expected = float(raw["graph_final_objective"])
            if not math.isclose(old_cost, expected, rel_tol=1e-7, abs_tol=1e-5):
                raise ValueError(
                    "Rebuilt robust graph objective differs from original: "
                    f"{old_cost} vs {expected}"
                )
            history = []
            stable = False
            for attempt in range(1, max_passes+1):
                solved, secs = solve(gtsam, graph, values)
                cost = float(graph.error(solved))
                if not math.isfinite(cost) or cost > old_cost + max(1e-7, abs(old_cost)*1e-10):
                    raise RuntimeError("Robust continuation objective invalid/increased")
                new_poses = poses_from_values(solved, n)
                shift = float(np.sqrt(np.mean(np.sum(
                    (new_poses[:, :3, 3]-poses[:, :3, 3])**2, axis=1
                ))))
                drop = max(0.0, (old_cost-cost)/max(abs(old_cost), 1.0))
                history.append({
                    "pass": attempt, "objective_before": old_cost,
                    "objective_after": cost, "relative_drop": drop,
                    "position_shift_rmse_m": shift, "seconds": secs,
                })
                print(f"  {spec['name']} pass={attempt} drop={drop:.3g} "
                      f"shift={shift:.6f}m", flush=True)
                values, poses, old_cost = solved, new_poses, cost
                if drop <= RTOL and shift <= SHIFT_TOL:
                    stable = True
                    break
            metrics = s7.evaluate(sequence, poses[:, :3, 3])
            saved = {
                "sequence": sequence, "spec": spec, "stage": "converge",
                "protocol": "uniform_robust_lm_continuation",
                "factor_counts": counts,
                "max_extra_passes": max_passes, "outer_relative_objective_tol": RTOL,
                "position_shift_rmse_tol_m": SHIFT_TOL,
                "stationary_under_rule": stable,
                "history": history,
                "final_metrics": metrics,
            }
            write_new_case(out, spec, poses, saved, stage="converge")
        ate = float(saved["final_metrics"]["ATE3D"])
        rows.append({
            "sequence": sequence, "case": spec["name"],
            "continued_ATE3D_m": ate,
            "stable": bool(saved["stationary_under_rule"]),
        })
        print(f"  {spec['name']:<34} ATE3D={ate:.6f}m "
              f"stable={saved['stationary_under_rule']}", flush=True)

    label = "_".join(s["name"] for s in specs)
    write_new_summary(out / f"isj_cross_date_robust_convergence_{label}.csv", rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    p.add_argument("--stage", choices=("fgo", "converge"), required=True)
    p.add_argument("--kernel", choices=("Huber", "Cauchy", "both"), default="both")
    p.add_argument("--scope", choices=("gps", "all", "both"), default="both")
    p.add_argument("--max-extra-passes", type=int, default=4)
    args = p.parse_args()
    date_for(args.sequence)
    if args.max_extra_passes < 1:
        p.error("--max-extra-passes must be >= 1")
    specs = specs_for(args.kernel, args.scope)
    out = validate_destination(args.sequence, RESULT_ROOT)
    with crossdate_bindings(args.sequence, RESULT_ROOT,
                            "converge" if args.stage == "converge" else "fgo"):
        if args.stage == "fgo":
            run_fgo(args.sequence, out, specs)
        else:
            run_converge(args.sequence, out, specs, args.max_extra_passes)


if __name__ == "__main__":
    main()
