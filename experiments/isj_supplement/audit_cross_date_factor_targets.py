from __future__ import annotations

"""READ-ONLY 0016 target-consistency audit against V3-style factor-error labels.

Reconstructs the SAME label *formula* as V3 on the frozen cross-date
sensor factors, using the existing FINAL_S0_fixed trajectory as the
rotation/pose reference. Compares GRU V5 and Mamba V6 predictions with
those reconstructed, target-frame-aligned labels, excluding t=0..62.

LIMITATIONS:
- FINAL_S0_fixed is a FGO solution formed from the measurements being
  diagnosed: it is NOT independent reference ground truth.
- Training used results/four_sensor_fixed_fg/poses.npy on 0027; this
  cross-date audit uses the local saved FINAL_S0_fixed analogue. The
  graph and trajectory provenance may differ, so comparisons assess
  a V3-style proxy, not an exactly reproduced or independent oracle.
- The GPS clean location derives from KITTI OXTS, same base source as
  degraded GPS input. This is a synthetic-fault diagnostic.
- U is Beta lack-of-evidence; no direct ground-truth uncertainty label
  or uncertainty calibration inference is attempted here.
- This drive has already been explored repeatedly: do not use the
  resulting metrics for tuning and then call 0016 untouched test data.

Does not modify any original predictions, checkpoints, factor graphs,
test results, or training data. Prints metrics only.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.oracle_factor_reliability import (
    OracleReliabilityConfig, compute_oracle_factor_reliability,
)
from experiments.isj_supplement.run_cross_date import (
    RESULT_ROOT, date_for, validate_destination, crossdate_bindings,
    result_is_complete, require_processed, prediction_preflight,
)
from experiments.isj_supplement.audit_cross_date_prediction_quality import (
    pearson, spearman,
)

WINDOW = 64
SENSORS = ("gps", "imu", "camera")
MODELS = ("gru", "mamba")
SOURCES = {"gru": "v5", "mamba": "v6"}


def quality_scores(predicted: np.ndarray, target: np.ndarray) -> dict:
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    true = np.asarray(target, dtype=float).reshape(-1)
    if pred.shape != true.shape or len(pred) < 3:
        raise ValueError("Predictions and targets must have equal length >= 3")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Predictions/targets contain NaN/Inf")
    err = pred - true
    baseline = np.full_like(true, true.mean())
    return {
        "n": len(true),
        "target_mean": float(true.mean()),
        "target_std": float(true.std()),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "mean_target_baseline_RMSE": float(np.sqrt(np.mean((baseline-true)**2))),
        "Pearson": float(pearson(pred, true)),
        "Spearman": float(spearman(pred, true)),
        "bias_pred_minus_target": float(err.mean()),
    }


def reconstruct_targets(out: Path, seq: str) -> tuple[dict, int]:
    from experiments.cross_sequence import run_stage7_factor_graph as s7

    n = require_processed(out)
    if not result_is_complete(out, "FINAL_S0_fixed", n):
        raise FileNotFoundError("Frozen FINAL_S0_fixed poses required")
    poses = np.load(out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy",
                    allow_pickle=False)
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Invalid reference pose matrices")

    measurements = s7.load_measurements(seq)
    sensor_file = out / "sensor_data.npz"
    with np.load(sensor_file, allow_pickle=False) as raw:
        clean = np.asarray(raw["clean_gps_local"], dtype=float)
    if clean.shape != (n, 3):
        raise ValueError("Invalid clean GPS shape")

    labels = compute_oracle_factor_reliability(
        clean_gps_local=clean,
        degraded_gps_local=measurements.gps_local,
        degraded_imu_gyro=measurements.imu_gyro,
        timestamps_seconds=measurements.timestamps,
        lidar_between=measurements.lidar_between,
        lidar_valid=measurements.lidar_valid,
        camera_between=measurements.camera_between,
        camera_valid=measurements.camera_valid,
        reference_poses=poses,
        config=OracleReliabilityConfig(),
    )
    expected = (
        "gps_reliability", "imu_reliability", "camera_reliability",
        "imu_rotation_error_deg",
    )
    for key in expected:
        vals = np.asarray(labels[key], dtype=float).reshape(-1)
        if vals.shape != (n,) or not np.isfinite(vals).all():
            raise ValueError(f"Invalid reconstructed {key}: shape={vals.shape}")
    return labels, n


def _f(num: float) -> str:
    return f"{num:+.4f}" if math.isfinite(num) else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    args = parser.parse_args()

    date_for(args.sequence)
    out = validate_destination(args.sequence, RESULT_ROOT)
    with crossdate_bindings(args.sequence, RESULT_ROOT, "fgo"):
        labels, n = reconstruct_targets(out, args.sequence)
        prediction_preflight(out, MODELS)

        start = WINDOW - 1
        print("=" * 110)
        print("READ-ONLY CROSS-DATE V3-STYLE FACTOR-ERROR TARGET AUDIT")
        print(f"sequence={args.sequence} n={n} evaluated_frames={n-start} (t={start}..{n-1})")
        print("Target = original OracleReliabilityConfig formula with saved local Fixed-FGO poses.")
        print("WARNING: Fixed-FGO pose reference is NOT independent truth; diagnostic only.")
        print("WARNING: V3 0027 reference graph and this 0016 Fixed-FGO may not be identical.")
        print("U is NOT tested for statistical calibration by these reliability metrics.")
        print("=" * 110)

        imu_error = np.asarray(labels["imu_rotation_error_deg"], float)[start:]
        imu_target = np.asarray(labels["imu_reliability"], float)[start:]
        print(
            f"IMU factor rotation mismatch (vs Fixed-FGO): "
            f"mean={imu_error.mean():.6f} deg; "
            f"p50={np.median(imu_error):.6f} deg; "
            f"p95={np.percentile(imu_error,95):.6f} deg"
        )
        print(
            f"IMU target distribution: mean={imu_target.mean():.6f}, "
            f"std={imu_target.std():.6f}, "
            f"min={imu_target.min():.6f}, "
            f"max={imu_target.max():.6f}"
        )

        from experiments.cross_sequence import run_stage7_factor_graph as s7
        for sensor in SENSORS:
            target = np.asarray(labels[f"{sensor}_reliability"], float)[start:]
            print(f"\n=== {sensor.upper()}: reconstructed reference target ===")
            print(f"  target: mean={target.mean():.4f}, std={target.std():.4f}, "
                  f"p05={np.percentile(target,5):.4f}, "
                  f"p95={np.percentile(target,95):.4f}")
            for model in MODELS:
                source = SOURCES[model]
                r, u = s7.pred(args.sequence, source, sensor, n)
                r = r[start:]
                u = u[start:]
                stats = quality_scores(r, target)
                print(
                    f"  {model.upper():6} "
                    f"r_MAE={stats['MAE']:.4f} "
                    f"r_RMSE={stats['RMSE']:.4f} "
                    f"target-mean_RMSE={stats['mean_target_baseline_RMSE']:.4f} "
                    f"Pearson={_f(stats['Pearson'])} "
                    f"Spearman={_f(stats['Spearman'])} "
                    f"mean_pred={stats['pred_mean']:.4f} "
                    f"bias={stats['bias_pred_minus_target']:+.4f} "
                    f"mean_U={u.mean():.4f}"
                )
        print("\nAUDIT COMPLETE (no files written)")


if __name__ == "__main__":
    main()
