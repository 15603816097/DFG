from __future__ import annotations

"""Read-only reconstruction/audit of Stage-7 *fixed-covariance* factor graphs.

Purpose:
  Explain why saved Stage-1 Fixed FGO and an identical Stage-2 Fixed FGO
  differ on drive 0042. This checks the ACTUAL graph objective, not ATE alone.

Reconstructs the original s7.run_case() fixed-only graph term-for-term, using
its exact GTSAM Pose3 parameterization, OXTS pose-0 prior, and noise anchors.
Does not need KITTI Raw, PyTorch, or mamba_ssm. It never edits Stage-7 results.

Optional --refine independently runs 100 LM steps from each saved solution,
reports graph cost, ATE and delta from initialization; only writes audit JSON.
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

import experiments.cross_sequence.run_stage7_factor_graph as s7
from experiments.cross_sequence.stage7_config import (
    TEST_SEQUENCES, sequence_out, GPS_FIXED, IMU_R0,
    LIDAR_R0, LIDAR_T0, CAM_R0, CAM_T0,
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


def graph_fixed(gtsam, m):
    """Build the original fixed Stage-7 graph with NO learned predictions."""
    n = len(m.gps_local)
    if n < 2:
        raise ValueError(f"Invalid sequence length {n}")
    cfg = FourSensorFactorConfig()
    graph = gtsam.NonlinearFactorGraph()
    anchor = s7.initial_pose_from_oxts(gtsam, m)
    graph.add(gtsam.PriorFactorPose3(
        0, anchor,
        gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray([cfg.prior_rotation_sigma] * 3 +
                       [cfg.prior_translation_sigma] * 3, dtype=float)),
    ))
    gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray([1e6] * 3 + [GPS_FIXED] * 3, dtype=float)
    )
    imu_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray([IMU_R0] * 3 + [cfg.imu_translation_sigma] * 3, dtype=float)
    )
    lidar_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray([LIDAR_R0] * 3 + [LIDAR_T0] * 3, dtype=float)
    )
    camera_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray([CAM_R0] * 3 + [CAM_T0] * 3, dtype=float)
    )

    ts = np.asarray(m.timestamps, dtype=float)
    counts = {"anchor": 1, "gps": 0, "imu": 0, "lidar": 0, "camera": 0}
    for i in range(n):
        graph.add(gtsam.PriorFactorPose3(
            i,
            gtsam.Pose3(
                gtsam.Rot3(),
                gtsam.Point3(*map(float, m.gps_local[i, :3]))
            ),
            gps_noise
        ))
        counts["gps"] += 1
        if i == n - 1:
            continue
        dt = float(ts[i + 1] - ts[i])
        if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            dt = 0.1
        graph.add(gtsam.BetweenFactorPose3(
            i, i + 1, s7._imu_between_pose(gtsam, m.imu_gyro[i], dt),
            imu_noise
        ))
        counts["imu"] += 1
        if bool(m.lidar_valid[i]):
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1, s7._pose3_from_matrix(gtsam, m.lidar_between[i]),
                lidar_noise
            ))
            counts["lidar"] += 1
        if bool(m.camera_valid[i]):
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1, s7._pose3_from_matrix(gtsam, m.camera_between[i]),
                camera_noise
            ))
            counts["camera"] += 1

    if graph.size() != sum(counts.values()):
        raise AssertionError(f"graph.size()={graph.size()} differs from {counts}")
    return graph, counts


def poses_to_values(gtsam, poses):
    values = gtsam.Values()
    for i, pose in enumerate(poses):
        values.insert(i, s7._pose3_from_matrix(gtsam, pose))
    return values


def values_to_poses(gtsam, values, n):
    return np.stack([
        s7._pose3_to_matrix(values.atPose3(i)) for i in range(n)
    ])


def numeric_delta(first, second):
    if first.shape != second.shape:
        raise ValueError("Pose shapes differ")
    d = second[:, :3, 3] - first[:, :3, 3]
    return {
        "position_change_rmse_m": float(np.sqrt(np.mean(np.sum(d * d, axis=1)))),
        "position_change_max_m": float(np.max(np.linalg.norm(d, axis=1))),
    }


def load_stage_data(seq):
    out = sequence_out(seq)
    paths = {
        "fixed1": out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy",
        "fixed2": out / "factor_graph" / "ISJ_CTRL_FIXED_SECOND_PASS" / "poses.npy",
        "reference_csv": out / "stage7_two_stage_results.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; first complete run_stage7_two_stage.py "
                "and run_controls.py --controls fixed2."
            )
    m = s7.load_measurements(seq)
    n = len(m.gps_local)
    saved = {}
    for stage in ("fixed1", "fixed2"):
        poses = np.load(paths[stage], allow_pickle=False)
        if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(f"{stage}: invalid poses shape/values {poses.shape}")
        saved[stage] = poses
    with paths["reference_csv"].open(newline="", encoding="utf-8") as stream:
        old = {r["case"]: r for r in csv.DictReader(stream)}
    if "S0_fixed" not in old:
        raise ValueError(f"No S0_fixed case in {paths['reference_csv']}")
    return m, saved, float(old["S0_fixed"]["ATE3D"])


def optimize_from(gtsam, graph, initial):
    # Mirrors the original s7.run_case() LM iteration budget and tolerance.
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-7)
    started = time.perf_counter()
    result = gtsam.LevenbergMarquardtOptimizer(
        graph, initial, params
    ).optimize()
    return result, time.perf_counter() - started


def run_audit(seq, refine=False):
    import gtsam

    m, saved, saved_ref_ate = load_stage_data(seq)
    n = len(m.gps_local)
    graph, counts = graph_fixed(gtsam, m)

    summary = {
        "sequence": seq,
        "classification": "read_only_post_hoc_convergence_audit",
        "graph_reconstructed_from": "s7.run_case fixed-only terms",
        "same_objective_for_fixed1_and_fixed2": True,
        "solver": "GTSAM LM",
        "solver_max_iterations": 100,
        "solver_relative_error_tolerance": 1e-7,
        "factor_counts": counts,
        "graph_factor_count": int(graph.size()),
        "saved": {},
    }

    initial_pose = s7.initial_pose_from_oxts(gtsam, m)
    initial_poses = s7.initial_trajectory_from_pose(gtsam, m, initial_pose)
    initial = gtsam.Values()
    for i, pose in enumerate(initial_poses):
        initial.insert(i, pose)
    summary["odometry_initialization"] = {
        "graph_objective": float(graph.error(initial)),
    }

    for stage in ("fixed1", "fixed2"):
        poses = saved[stage]
        values = poses_to_values(gtsam, poses)
        error = float(graph.error(values))
        metrics = s7.evaluate(seq, poses[:, :3, 3])
        if not math.isfinite(error) or error < 0:
            raise ValueError(f"{seq} {stage}: invalid objective {error}")
        summary["saved"][stage] = {
            "graph_objective": error,
            "ATE3D": float(metrics["ATE3D"]),
            "ATE2D": float(metrics["ATE2D"]),
            "ZRMSE": float(metrics["ZRMSE"]),
        }

    if abs(summary["saved"]["fixed1"]["ATE3D"] - saved_ref_ate) > 1e-4:
        raise ValueError(
            f"{seq}: saved fixed1 poses yield different ATE than CSV; "
            "possible mix of experiment versions. Audit halted."
        )

    e1 = summary["saved"]["fixed1"]["graph_objective"]
    e2 = summary["saved"]["fixed2"]["graph_objective"]
    summary["fixed2_graph_reduction_pct"] = (
        100.0 * (e1 - e2) / e1 if e1 > 0 else None
    )
    summary["fixed2_ate_improvement_pct"] = (
        100.0 * (
            summary["saved"]["fixed1"]["ATE3D"] -
            summary["saved"]["fixed2"]["ATE3D"]
        ) / summary["saved"]["fixed1"]["ATE3D"]
    )
    summary["fixed1_to_fixed2"] = numeric_delta(saved["fixed1"], saved["fixed2"])

    if refine:
        summary["refinement"] = {}
        for stage in ("fixed1", "fixed2"):
            values = poses_to_values(gtsam, saved[stage])
            result, seconds = optimize_from(gtsam, graph, values)
            new_poses = values_to_poses(gtsam, result, n)
            final_error = float(graph.error(result))
            summary["refinement"][stage] = {
                "seconds": seconds,
                "graph_objective_start": float(graph.error(values)),
                "graph_objective_end": final_error,
                "ATE3D": s7.evaluate(seq, new_poses[:, :3, 3])["ATE3D"],
                **numeric_delta(saved[stage], new_poses),
            }

    out = sequence_out(seq) / "isj_fixed_convergence_audit.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 84}\n{seq}  |  factor counts: {counts}")
    print(f"initial objective     = {summary['odometry_initialization']['graph_objective']:.9f}")
    for stage in ("fixed1", "fixed2"):
        row = summary["saved"][stage]
        print(f"{stage:19} objective={row['graph_objective']:.9f} "
              f"ATE3D={row['ATE3D']:.6f}m")
    print(f"fixed2 objective reduction = {summary['fixed2_graph_reduction_pct']:.5f}%")
    print(f"fixed2 ATE improvement      = {summary['fixed2_ate_improvement_pct']:.5f}%")
    print(f"fixed1 -> fixed2 pose RMSE  = "
          f"{summary['fixed1_to_fixed2']['position_change_rmse_m']:.6f}m")
    if refine:
        for stage, row in summary["refinement"].items():
            print(f"Re-optimizing {stage}: "
                  f"objective {row['graph_objective_start']:.9f}"
                  f" -> {row['graph_objective_end']:.9f}, "
                  f"ATE3D={row['ATE3D']:.6f}m, elapsed={row['seconds']:.2f}s")
    print("Saved diagnostic JSON:", out)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", choices=("all",) + TEST_SEQUENCES, default="all")
    p.add_argument("--refine", action="store_true",
                   help="Also run extra LM passes for convergence diagnosis.")
    args = p.parse_args()
    sequences = TEST_SEQUENCES if args.sequence == "all" else (args.sequence,)
    for sequence in sequences:
        run_audit(sequence, refine=args.refine)


if __name__ == "__main__":
    main()
