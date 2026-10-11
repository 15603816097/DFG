from __future__ import annotations

"""Post-hoc Stage-7 uncertainty ablation with frozen model outputs.

Each method uses the EXACT same fixed-FG warm-start, sensor factors,
hyperparameters, and pretrained predictions as the original two-stage study.

Direct:          sigma = reliability_to_sigma(r)
Uncertainty-aware: sigma = sigma_fixed + (1-u)*(reliability_to_sigma(r)-sigma_fixed)

This script does not train or tune. It does not overwrite the original
stage7_results.csv or stage7_two_stage_results.csv. Because this ablation
was devised after inspecting 0042, it is exploratory/post-hoc, NOT a new
independent blind test. Results are stored in separate ABL_* directories.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, choices=TEST_SEQUENCES)
    args = parser.parse_args()

    seq = args.sequence
    od = sequence_out(seq)
    fixed_path = od / "factor_graph" / "FINAL_S0_fixed" / "poses.npy"
    existing_path = od / "stage7_two_stage_results.csv"

    if not fixed_path.is_file() or not existing_path.is_file():
        raise FileNotFoundError(
            "Run run_stage7_two_stage.py first; this ablation reuses its "
            "fixed-FG initial values and uncertainty-aware reference metrics."
        )

    m = s7.load_measurements(seq)
    n = len(m.gps_local)
    fixed_poses = np.load(fixed_path, allow_pickle=False)
    if fixed_poses.shape != (n, 4, 4) or not np.isfinite(fixed_poses).all():
        raise ValueError(f"Invalid fixed poses: {fixed_poses.shape}")

    with existing_path.open("r", newline="", encoding="utf-8") as stream:
        reference = {row["case"]: row for row in csv.DictReader(stream)}

    required = ("S0_fixed", "S1_v5_gru", "S2_v6_all_mamba")
    missing = [name for name in required if name not in reference]
    if missing:
        raise ValueError(f"Missing two-stage reference cases: {missing}")

    original_init = s7.initial_trajectory_from_pose
    original_confidence = s7.confidence

    def frozen_fixed_initialization(gtsam, measurements, initial_pose):
        if len(measurements.gps_local) != n:
            raise ValueError("Sequence length mismatch")
        return [s7._pose3_from_matrix(gtsam, T) for T in fixed_poses]

    # Only change vs the official dynamic method: set (1-u) = 1.
    def direct_reliability_confidence(unused_uncertainty):
        return 1.0

    rows = []
    started = time.perf_counter()
    try:
        s7.initial_trajectory_from_pose = frozen_fixed_initialization
        s7.confidence = direct_reliability_confidence

        for method, source in (("V5_GRU", "v5"), ("V6_MAMBA", "v6")):
            print(f"Running reliability-only ablation: {seq} {method}", flush=True)
            metrics = s7.run_case(
                seq=seq,
                name=f"ABL_{method}_RELIABILITY_ONLY",
                gps_src=source,
                imu_src=source,
                cam_src=source,
            )
            ref_case = "S1_v5_gru" if source == "v5" else "S2_v6_all_mamba"
            with_uncertainty = float(reference[ref_case]["ATE3D"])
            without_uncertainty = float(metrics["ATE3D"])
            fixed = float(reference["S0_fixed"]["ATE3D"])
            rows.append({
                "sequence": seq,
                "method": method,
                "fixed_ATE3D": fixed,
                "reliability_only_ATE3D": without_uncertainty,
                "reliability_plus_uncertainty_ATE3D": with_uncertainty,
                "uncertainty_gain_pct_vs_reliability_only": (
                    100.0 * (without_uncertainty - with_uncertainty)
                    / without_uncertainty
                ),
                "reliability_only_gain_pct_vs_fixed": (
                    100.0 * (fixed - without_uncertainty) / fixed
                ),
                "uncertainty_aware_gain_pct_vs_fixed": (
                    100.0 * (fixed - with_uncertainty) / fixed
                ),
            })
    finally:
        s7.confidence = original_confidence
        s7.initial_trajectory_from_pose = original_init

    report = od / "stage7_uncertainty_ablation.csv"
    with report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    protocol = od / "stage7_uncertainty_ablation_protocol.json"
    protocol.write_text(json.dumps({
        "status": "exploratory_post_hoc",
        "origin": "ablation planned after Stage-7 results inspected",
        "model_training": False,
        "prediction_recalculation": False,
        "warm_start": "FINAL_S0_fixed frozen poses",
        "original_uncertainty_aware": "1-u confidence shrinkage",
        "ablation": "set confidence=1; reliability-only mapping",
        "all_sensor_sources": "GPS, IMU and Camera use same V5 or V6",
        "lidar_sigma": "fixed",
        "ground_truth_used_for_initialization": False,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }, indent=2), encoding="utf-8")

    print("\nSTAGE 7 UNCERTAINTY ABLATION COMPLETE")
    for row in rows:
        print(
            f"{row['method']:9s} "
            f"direct={row['reliability_only_ATE3D']:.6f} "
            f"uncertainty={row['reliability_plus_uncertainty_ATE3D']:.6f} "
            f"uncertainty_gain={row['uncertainty_gain_pct_vs_reliability_only']:+.2f}%"
        )
    print(f"Saved: {report}")
    print("Original Stage-7 result files: NOT MODIFIED")


if __name__ == "__main__":
    main()
