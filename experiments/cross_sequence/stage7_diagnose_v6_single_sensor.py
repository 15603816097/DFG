from __future__ import annotations

"""Frozen V6 single-sensor Stage-7 diagnostic; does not alter official results."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence.run_stage7_factor_graph import run_case
from experiments.cross_sequence.stage7_config import TEST_SEQUENCES, sequence_out


DIAGNOSTIC_CASES = (
    ("DIAG_v6_gps_only", "v6", None, None),
    ("DIAG_v6_imu_only", None, "v6", None),
    ("DIAG_v6_camera_only", None, None, "v6"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify V6 sensor driving Stage-7 cross-sequence error."
    )
    parser.add_argument("--sequence", required=True, choices=TEST_SEQUENCES)
    args = parser.parse_args()

    output_dir = sequence_out(args.sequence)
    baseline_path = output_dir / "factor_graph" / "S0_fixed" / "metrics.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Corrected S0 baseline missing: {baseline_path}"
        )

    with baseline_path.open("r", encoding="utf-8") as file:
        baseline = json.load(file)
    fixed_ate = float(baseline["ATE3D"])
    if not (0 < fixed_ate < float("inf")):
        raise ValueError(f"Invalid fixed baseline ATE3D: {fixed_ate}")

    print(f"Sequence: {args.sequence}", flush=True)
    print(f"Fixed baseline ATE3D: {fixed_ate:.6f} m", flush=True)
    print("Frozen V6 predictions; other sensors fixed; no training.", flush=True)

    rows = [{
        "case": "S0_fixed_existing",
        "ATE3D": fixed_ate,
        "ATE2D": baseline.get("ATE2D"),
        "ZRMSE": baseline.get("ZRMSE"),
        "vs_fixed_pct": 0.0,
    }]

    for name, gps_source, imu_source, camera_source in DIAGNOSTIC_CASES:
        metrics = run_case(
            args.sequence,
            name,
            gps_src=gps_source,
            imu_src=imu_source,
            cam_src=camera_source,
        )
        ate = float(metrics["ATE3D"])
        delta = 100.0 * (fixed_ate - ate) / fixed_ate
        rows.append({
            "case": name,
            "ATE3D": ate,
            "ATE2D": float(metrics["ATE2D"]),
            "ZRMSE": float(metrics["ZRMSE"]),
            "vs_fixed_pct": delta,
        })
        print(f"{name}: ATE3D={ate:.6f} m, vs fixed={delta:+.2f}%", flush=True)

    report_path = output_dir / "stage7_v6_single_sensor_diagnostic.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case", "ATE3D", "ATE2D", "ZRMSE", "vs_fixed_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("DIAGNOSTIC COMPLETE", flush=True)
    print(f"Saved: {report_path}", flush=True)
    print("Official stage7_results.csv: NOT MODIFIED", flush=True)


if __name__ == "__main__":
    main()
