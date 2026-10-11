from __future__ import annotations

"""
Stage-7 two-stage, frozen-model cross-sequence evaluation.

Stage 1: solve the fixed-covariance factor graph from the same LiDAR/Camera
         odometry initialization used by the original Stage-7 implementation.
Stage 2: initialize each dynamic-covariance factor graph from the *same*
         Stage-1 fixed solution. No test ground truth is used for initialization.

This separate runner does not overwrite original Stage-7 case directories or
stage7_results.csv. It uses the existing saved V5/V6 predictions and requires
the local Stage-7 runner to have the OXTS initialization correction.
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
from experiments.cross_sequence.stage7_config import TEST_SEQUENCES, sequence_out


DYNAMIC_CASES = (
    ("S1_v5_gru", "v5", "v5", "v5"),
    ("S2_v6_all_mamba", "v6", "v6", "v6"),
    ("S3_d5_hybrid", "v6", "v5", "v6"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen Stage-7 two-stage fixed-initialized FG evaluation."
    )
    parser.add_argument("--sequence", required=True, choices=TEST_SEQUENCES)
    args = parser.parse_args()
    seq = args.sequence
    out = sequence_out(seq)

    # Required to avoid silently reverting to the old identity yaw convention.
    measurements = s7.load_measurements(seq)
    if not hasattr(measurements, "initial_oxts_rpy"):
        raise RuntimeError("Stage-7 loader lacks OXTS initial attitude.")
    if not hasattr(s7, "initial_trajectory_from_pose"):
        raise RuntimeError("Local Stage-7 FG runner must include OXTS initialization fix.")

    print("=" * 78, flush=True)
    print("STAGE 7 / TWO-STAGE FACTOR GRAPH", flush=True)
    print(f"Sequence: {seq}", flush=True)
    print("Frozen predictors, original covariance mapping, no GT initialization.", flush=True)
    print("Stage 1 = fixed covariance; Stage 2 = fixed-solution warm start.", flush=True)
    print("=" * 78, flush=True)

    rows = []
    start = time.perf_counter()
    fixed_metrics = s7.run_case(seq=seq, name="FINAL_S0_fixed")
    fixed_seconds = time.perf_counter() - start
    fixed_ate = float(fixed_metrics["ATE3D"])
    if not np.isfinite(fixed_ate) or fixed_ate <= 0:
        raise RuntimeError(f"Invalid fixed ATE3D: {fixed_ate}")

    rows.append({
        "case": "S0_fixed",
        **fixed_metrics,
        "vs_fixed_pct": 0.0,
        "solver_seconds": round(fixed_seconds, 3),
    })

    fixed_pose_path = out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy"
    fixed_poses = np.load(fixed_pose_path, allow_pickle=False)
    n = len(measurements.gps_local)
    if fixed_poses.shape != (n, 4, 4) or not np.isfinite(fixed_poses).all():
        raise ValueError(f"Invalid fixed poses shape/values: {fixed_poses.shape}")

    original_initializer = s7.initial_trajectory_from_pose

    def fixed_initialization(gtsam, m, initial_pose):
        if len(m.gps_local) != len(fixed_poses):
            raise ValueError("Fixed poses do not match current sequence length")
        return [s7._pose3_from_matrix(gtsam, transform) for transform in fixed_poses]

    try:
        s7.initial_trajectory_from_pose = fixed_initialization
        for name, gps_source, imu_source, camera_source in DYNAMIC_CASES:
            print(f"\nRunning two-stage {name} ...", flush=True)
            started = time.perf_counter()
            metrics = s7.run_case(
                seq=seq,
                name="FINAL_" + name,
                gps_src=gps_source,
                imu_src=imu_source,
                cam_src=camera_source,
            )
            seconds = time.perf_counter() - started
            ate = float(metrics["ATE3D"])
            rows.append({
                "case": name,
                **metrics,
                "vs_fixed_pct": (fixed_ate - ate) / fixed_ate * 100.0,
                "solver_seconds": round(seconds, 3),
            })
    finally:
        s7.initial_trajectory_from_pose = original_initializer

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "stage7_two_stage_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    protocol_path = out / "stage7_two_stage_protocol.json"
    protocol_path.write_text(json.dumps({
        "sequence": seq,
        "protocol": "two_stage_fixed_warmstart",
        "first_stage": "Fixed covariance FG, odometry-initialized",
        "second_stage": "V5/V6/Hybrid dynamic FG, initialized from same fixed solution",
        "initial_attitude_source": "OXTS frame 0 RPY",
        "ground_truth_used_for_initialization": False,
        "training_on_test_sequence": False,
        "hyperparameter_tuning_on_test_sequence": False,
        "fixed_stage_seconds": round(fixed_seconds, 3),
        "case_run_times_seconds": {
            row["case"]: row["solver_seconds"] for row in rows
        },
        "outputs": str(csv_path),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("STAGE 7 TWO-STAGE RESULTS")
    for row in rows:
        print(
            f"{row['case']:18s} "
            f"ATE3D={row['ATE3D']:.6f} "
            f"ATE2D={row['ATE2D']:.6f} "
            f"ZRMSE={row['ZRMSE']:.6f} "
            f"vs_fixed={row['vs_fixed_pct']:+.2f}% "
            f"solver={row['solver_seconds']:.1f}s"
        )
    print(f"Results: {csv_path}")
    print(f"Protocol: {protocol_path}")
    print("Original stage7_results.csv: NOT MODIFIED")
    print("STAGE 7 TWO-STAGE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
