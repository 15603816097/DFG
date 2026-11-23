from __future__ import annotations

"""Read-only Stage-7 learned dynamic-FGO convergence audit.

Reconstructs the frozen Stage-7 graph for V5 GRU and V6 Mamba using the
SAME saved model reliability/uncertainty files and graph/noise semantics as
run_stage7_factor_graph.py. Compares graph objective at the common fixed-FGO
warm start with objective at saved dynamic solution, and optionally runs
another 100 unchanged LM iterations from the saved dynamic solution.

No model training, no KITTI Raw needed, no original results overwritten.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.cross_sequence.run_stage7_factor_graph as s7
from experiments.cross_sequence.stage7_config import (
    TEST_SEQUENCES, sequence_out, LIDAR_R0, LIDAR_T0,
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig

CASES = (
    ("S1_v5_gru", "v5"),
    ("S2_v6_all_mamba", "v6"),
)


def build_dynamic_graph(gtsam, seq, m, source):
    """Match s7.run_case graph with gps_src=imu_src=cam_src=source."""
    n = len(m.gps_local)
    cfg = FourSensorFactorConfig()
    cached = {
        sensor: s7.pred(seq, source, sensor, n)
        for sensor in ("gps", "imu", "camera")
    }
    graph = gtsam.NonlinearFactorGraph()
    anchor = s7.initial_pose_from_oxts(gtsam, m)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(
        [cfg.prior_rotation_sigma] * 3 +
        [cfg.prior_translation_sigma] * 3, dtype=float))
    graph.add(gtsam.PriorFactorPose3(0, anchor, prior_noise))

    times = np.asarray(m.timestamps, dtype=float)
    for i in range(n):
        rg, ug = cached["gps"]
        gs = s7.gps_sigma(cfg, rg[i], ug[i])
        gps_pose = gtsam.Pose3(
            gtsam.Rot3(),
            gtsam.Point3(*map(float, m.gps_local[i, :3]))
        )
        graph.add(gtsam.PriorFactorPose3(
            i, gps_pose,
            gtsam.noiseModel.Diagonal.Sigmas(np.asarray(
                [1e6] * 3 + [gs] * 3, dtype=float)),
        ))
        if i == n - 1:
            continue

        dt = float(times[i + 1] - times[i])
        if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            dt = 0.1
        ri, ui = cached["imu"]
        ir = s7.imu_sigma(cfg, ri[i + 1], ui[i + 1])
        graph.add(gtsam.BetweenFactorPose3(
            i, i + 1,
            s7._imu_between_pose(gtsam, m.imu_gyro[i], dt),
            gtsam.noiseModel.Diagonal.Sigmas(np.asarray(
                [ir] * 3 + [cfg.imu_translation_sigma] * 3, dtype=float)),
        ))

        if bool(m.lidar_valid[i]):
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1,
                s7._pose3_from_matrix(gtsam, m.lidar_between[i]),
                gtsam.noiseModel.Diagonal.Sigmas(np.asarray(
                    [LIDAR_R0] * 3 + [LIDAR_T0] * 3, dtype=float)),
            ))

        if bool(m.camera_valid[i]):
            rc, uc = cached["camera"]
            cr, ct = s7.cam_sigma(cfg, rc[i + 1], uc[i + 1])
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1,
                s7._pose3_from_matrix(gtsam, m.camera_between[i]),
                gtsam.noiseModel.Diagonal.Sigmas(np.asarray(
                    [cr] * 3 + [ct] * 3, dtype=float)),
            ))
    return graph


def values_from_poses(gtsam, poses):
    vals = gtsam.Values()
    for i, pose in enumerate(poses):
        vals.insert(i, s7._pose3_from_matrix(gtsam, pose))
    return vals


def poses_from_values(values, n):
    return np.stack([s7._pose3_to_matrix(values.atPose3(i))
                     for i in range(n)])


def check_saved_pose(path, n):
    if not path.is_file():
        raise FileNotFoundError(path)
    poses = np.load(path, allow_pickle=False)
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"{path}: expected finite ({n},4,4) poses, got {poses.shape}")
    return poses


def evaluate_one(gtsam, seq, m, first, model_case, source, reference, refine):
    out = sequence_out(seq)
    n = len(m.gps_local)
    model_path = out / "factor_graph" / ("FINAL_" + model_case) / "poses.npy"
    saved = check_saved_pose(model_path, n)

    graph = build_dynamic_graph(gtsam, seq, m, source)
    v_first = values_from_poses(gtsam, first)
    v_saved = values_from_poses(gtsam, saved)

    initial_cost = float(graph.error(v_first))
    saved_cost = float(graph.error(v_saved))
    saved_ate = float(s7.evaluate(seq, saved[:, :3, 3])["ATE3D"])
    expected = float(reference[model_case]["ATE3D"])
    if abs(expected - saved_ate) > 1e-4:
        raise ValueError(
            f"{seq} {model_case}: pose/CSV ATE mismatch "
            f"({saved_ate:.9f} vs {expected:.9f}); possibly mixed result versions"
        )
    if not np.isfinite([initial_cost, saved_cost]).all():
        raise ValueError("Invalid cost")

    row = {
        "model": model_case,
        "source": source,
        "graph_factors": int(graph.size()),
        "warm_start_objective": initial_cost,
        "saved_dynamic_objective": saved_cost,
        "saved_dynamic_ATE3D": saved_ate,
        "warm_start_to_saved_objective_reduction_pct":
            100.0 * (initial_cost - saved_cost) / max(initial_cost, 1e-12),
    }

    if refine:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        params.setRelativeErrorTol(1e-7)
        started = time.perf_counter()
        refined = gtsam.LevenbergMarquardtOptimizer(
            graph, v_saved, params
        ).optimize()
        seconds = time.perf_counter() - started
        refined_poses = poses_from_values(refined, n)
        residual_displacement = refined_poses[:, :3, 3] - saved[:, :3, 3]
        row["refinement"] = {
            "solver_seconds": seconds,
            "objective": float(graph.error(refined)),
            "ATE3D": float(s7.evaluate(seq, refined_poses[:, :3, 3])["ATE3D"]),
            "position_shift_rmse_m": float(np.sqrt(np.mean(
                np.sum(residual_displacement ** 2, axis=1)))),
        }

    print(f"  {model_case}: warm={initial_cost:.6f}, "
          f"saved={saved_cost:.6f}, saved ATE3D={saved_ate:.6f}m")
    if refine:
        rr = row["refinement"]
        print(f"       continuation objective={rr['objective']:.6f}, "
              f"ATE3D={rr['ATE3D']:.6f}m, "
              f"pose RMSE shift={rr['position_shift_rmse_m']:.6f}m")
    return row


def audit(seq, refine):
    import gtsam

    out = sequence_out(seq)
    m = s7.load_measurements(seq)
    n = len(m.gps_local)
    fixed_path = out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy"
    fixed = check_saved_pose(fixed_path, n)
    csv_path = out / "stage7_two_stage_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reference = {r["case"]: r for r in csv.DictReader(stream)}
    for case, _source in CASES:
        if case not in reference:
            raise KeyError(f"Missing {case} in {csv_path}")

    print(f"\n{'=' * 80}\n{seq}: dynamic FGO convergence")
    rows = [
        evaluate_one(gtsam, seq, m, fixed, case, source, reference, refine)
        for case, source in CASES
    ]
    summary = {
        "sequence": seq,
        "kind": "read_only_post_hoc_dynamic_fgo_convergence_audit",
        "protocol": "reconstructed_original_graph_and_frozen_predictions",
        "maxLMIterations": 100,
        "relativeErrorTol": 1e-7,
        "notes": [
            "All saved dynamic outputs are compared to their own reconstructed graph, not to the fixed graph.",
            "Refinement is one more 100-iteration budget from saved dynamic estimate.",
            "Unchanged objective on re-optimization is not proof of global optimality.",
        ],
        "results": rows,
    }
    path = out / "isj_dynamic_convergence_audit.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("all",) + TEST_SEQUENCES, default="all")
    parser.add_argument("--refine", action="store_true",
                        help="Run an additional 100 LM iterations from each saved dynamic trajectory")
    args = parser.parse_args()
    sequences = TEST_SEQUENCES if args.sequence == "all" else (args.sequence,)
    for seq in sequences:
        audit(seq, args.refine)


if __name__ == "__main__":
    main()
