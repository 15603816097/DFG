from __future__ import annotations

"""Exploratory fixed-covariance Huber/Cauchy GPS robustness baselines.

Reproduce the Stage-7 pose graph's fixed IMU/LiDAR/Camera/GPS factors,
except robustify the GPS measurement priors with a standard M-estimator.
Both methods use the SAME saved Stage-1 fixed-FG solution as optimization
initialization. No trained model, GT-based initialization, parameter search,
or modification of existing Stage-7 experiment outputs.

This experiment was added after observing Stage-7 results. It should not
be presented as a prospectively designed independent blind test.
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
    TEST_SEQUENCES,
    sequence_out,
    GPS_FIXED,
    IMU_R0,
    LIDAR_R0,
    LIDAR_T0,
    CAM_R0,
    CAM_T0,
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


# Frozen conventional robust-kernel settings; never tuned on Stage-7 sequences.
KERNELS = (
    ("Huber", 1.345),
    ("Cauchy", 2.3849),
)


def run_one(sequence: str, kernel: str, threshold: float, fixed_poses: np.ndarray):
    import gtsam

    m = s7.load_measurements(sequence)
    n = len(m.gps_local)
    if fixed_poses.shape != (n, 4, 4):
        raise ValueError(f"{sequence}: fixed-pose shape mismatch {fixed_poses.shape}")

    cfg = FourSensorFactorConfig()
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    anchor = s7.initial_pose_from_oxts(gtsam, m)
    prior_sigmas = np.array(
        [cfg.prior_rotation_sigma] * 3 + [cfg.prior_translation_sigma] * 3,
        dtype=np.float64,
    )
    graph.add(gtsam.PriorFactorPose3(
        0, anchor, gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
    ))

    for i in range(n):
        initial.insert(i, s7._pose3_from_matrix(gtsam, fixed_poses[i]))

    base_gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e6] * 3 + [GPS_FIXED] * 3, dtype=np.float64)
    )
    estimator_class = getattr(gtsam.noiseModel.mEstimator, kernel)
    robust_gps_noise = gtsam.noiseModel.Robust.Create(
        estimator_class.Create(threshold), base_gps_noise
    )

    imu_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([IMU_R0] * 3 + [cfg.imu_translation_sigma] * 3, dtype=np.float64)
    )
    lidar_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([LIDAR_R0] * 3 + [LIDAR_T0] * 3, dtype=np.float64)
    )
    camera_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([CAM_R0] * 3 + [CAM_T0] * 3, dtype=np.float64)
    )

    times = np.asarray(m.timestamps, dtype=np.float64)
    counts = {"gps": 0, "imu": 0, "lidar": 0, "camera": 0}

    for i in range(n):
        graph.add(gtsam.PriorFactorPose3(
            i,
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(
                *map(float, m.gps_local[i, :3])
            )),
            robust_gps_noise,
        ))
        counts["gps"] += 1

        if i == n - 1:
            continue

        dt = float(times[i + 1] - times[i])
        if not np.isfinite(dt) or dt <= 0 or dt > 1.0:
            dt = 0.1

        graph.add(gtsam.BetweenFactorPose3(
            i, i + 1, s7._imu_between_pose(gtsam, m.imu_gyro[i], dt),
            imu_noise,
        ))
        counts["imu"] += 1

        if bool(m.lidar_valid[i]):
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1, s7._pose3_from_matrix(gtsam, m.lidar_between[i]),
                lidar_noise,
            ))
            counts["lidar"] += 1

        if bool(m.camera_valid[i]):
            graph.add(gtsam.BetweenFactorPose3(
                i, i + 1, s7._pose3_from_matrix(gtsam, m.camera_between[i]),
                camera_noise,
            ))
            counts["camera"] += 1

    print(f"Sequence={sequence} kernel={kernel} k={threshold}")
    print(f"Frames={n}; factor_counts={counts}")
    initial_objective = float(graph.error(initial))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-7)

    t0 = time.perf_counter()
    result = gtsam.LevenbergMarquardtOptimizer(
        graph, initial, params
    ).optimize()
    elapsed = time.perf_counter() - t0
    final_objective = float(graph.error(result))

    poses = np.stack([
        s7._pose3_to_matrix(result.atPose3(i))
        for i in range(n)
    ])
    traj = poses[:, :3, 3]
    if not np.isfinite(poses).all():
        raise RuntimeError(f"Non-finite optimized trajectory: {kernel}")

    metrics = s7.evaluate(sequence, traj)
    result_dir = sequence_out(sequence) / "factor_graph" / f"ROBUST_GPS_{kernel.upper()}"
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "poses.npy", poses)
    np.savetxt(result_dir / "trajectory.txt", traj, fmt="%.9f")
    metadata = {
        "sequence": sequence,
        "kernel": kernel,
        "threshold": threshold,
        "applied_to": "GPS prior factors only",
        "all_other_factors": "same fixed covariances and validity checks as Stage-7 S0",
        "initialization": "saved FINAL_S0_fixed poses",
        "ground_truth_used_for_initialization": False,
        "initial_graph_objective": initial_objective,
        "final_graph_objective": final_objective,
        "solver_seconds": elapsed,
        "factor_counts": counts,
        "metrics": metrics,
        "status": "exploratory_post_hoc",
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"{kernel}: ATE3D={metrics['ATE3D']:.6f}m "
        f"graph_error={initial_objective:.3f}->{final_objective:.3f} "
        f"solver={elapsed:.2f}s"
    )
    return metrics, elapsed, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=TEST_SEQUENCES)
    args = parser.parse_args()
    sequence = args.sequence
    out = sequence_out(sequence)

    fixed_path = out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy"
    summary_path = out / "stage7_two_stage_results.csv"
    if not fixed_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Run run_stage7_two_stage.py first; fixed warm-start results are required."
        )

    fixed_poses = np.load(fixed_path, allow_pickle=False)
    if not np.isfinite(fixed_poses).all():
        raise ValueError("Non-finite Stage-1 poses")
    with summary_path.open("r", newline="", encoding="utf-8") as stream:
        previous = {r["case"]: r for r in csv.DictReader(stream)}
    for name in ("S0_fixed", "S1_v5_gru", "S2_v6_all_mamba"):
        if name not in previous:
            raise KeyError(f"Missing reference case {name}")

    fixed_ate = float(previous["S0_fixed"]["ATE3D"])
    rows = []
    for kind, threshold in KERNELS:
        metrics, seconds, counts = run_one(
            sequence, kind, threshold, fixed_poses
        )
        rows.append({
            "sequence": sequence,
            "case": f"ROBUST_GPS_{kind.upper()}",
            "ATE3D": metrics["ATE3D"],
            "ATE2D": metrics["ATE2D"],
            "ZRMSE": metrics["ZRMSE"],
            "vs_fixed_pct": 100 * (fixed_ate - metrics["ATE3D"]) / fixed_ate,
            "solver_seconds": round(seconds, 3),
        })

    report = out / "stage7_robust_gps_baselines.csv"
    with report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nROBUST GPS BASELINES COMPLETE")
    print("Fixed ATE3D:", fixed_ate)
    print("V5 uncertainty ATE3D:", float(previous["S1_v5_gru"]["ATE3D"]))
    print("V6 uncertainty ATE3D:", float(previous["S2_v6_all_mamba"]["ATE3D"]))
    for row in rows:
        print(f"{row['case']}: ATE3D={row['ATE3D']:.6f} vs Fixed={row['vs_fixed_pct']:+.2f}%")
    print("Saved:", report)
    print("Official Stage-7 outputs were NOT modified.")


if __name__ == "__main__":
    main()
