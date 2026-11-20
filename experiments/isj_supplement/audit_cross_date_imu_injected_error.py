from __future__ import annotations

"""READ-ONLY diagnostic of IMU injected error vs frozen r/U on cross-date KITTI.

Important alignment in Stage-7 run_stage7_factor_graph.py:
  Factor (i -> i+1) uses degraded_imu_gyro[i], but weight r[i+1],U[i+1].
Report BOTH same-frame and factor-aligned associations to distinguish a
one-step label/measurement alignment convention. These are correlations with
KNOWN APPLIED SYNTHETIC PERTURBATION, not IMU measurement accuracy vs an
independent physical inertial reference or validated network calibration.

No outputs, predictions, trained weights, graphs or source data are modified.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.isj_supplement.run_cross_date import (
    RESULT_ROOT, RAW_ROOT, date_for, validate_destination,
)
from experiments.isj_supplement.audit_cross_date_prediction_quality import (
    pearson, spearman,
)
from src.loader.imu_loader import IMULoader


WINDOW = 64
SENSORS = ("v5", "v6")


def injected_imu_errors(
    clean_gyro: np.ndarray,
    degraded_gyro: np.ndarray,
    clean_acc: np.ndarray,
    degraded_acc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c_g = np.asarray(clean_gyro, dtype=float)
    d_g = np.asarray(degraded_gyro, dtype=float)
    c_a = np.asarray(clean_acc, dtype=float)
    d_a = np.asarray(degraded_acc, dtype=float)
    if not (c_g.shape == d_g.shape == c_a.shape == d_a.shape):
        raise ValueError("Clean/degraded gyro and acceleration shapes differ")
    if c_g.ndim != 2 or c_g.shape[1] != 3 or not np.isfinite(
        np.stack((c_g,d_g,c_a,d_a))
    ).all():
        raise ValueError("Expected finite IMU arrays of shape (N,3)")
    gyro = np.linalg.norm(d_g - c_g, axis=1)
    acc = np.linalg.norm(d_a - c_a, axis=1)
    return gyro, acc


def aligned_series(reference_error: np.ndarray, r: np.ndarray,
                   u: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """offset=0 same t; offset=1 factor (i -> i+1) uses IMU[i], prediction[i+1]."""
    ref = np.asarray(reference_error, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    if len(ref) != len(r) or len(r) != len(u) or len(ref) < WINDOW + 1:
        raise ValueError("Expected aligned IMU and predictions >64 frames")
    if offset not in (0, 1):
        raise ValueError("offset must be 0 or 1")
    index = np.arange(WINDOW - 1, len(r), dtype=int)
    return ref[index - offset], 1.0 - r[index], u[index]


def association(a, b):
    x, y = np.asarray(a, float), np.asarray(b, float)
    p, s = pearson(x, y), spearman(x, y)
    def fmt(v):
        return "n/a" if not math.isfinite(v) else f"{v:+.4f}"
    return f"Pearson={fmt(p):>8s} Spearman={fmt(s):>8s}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    args = parser.parse_args()

    date = date_for(args.sequence)
    out = validate_destination(args.sequence, RESULT_ROOT)
    path = out / "sensor_data.npz"
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        needed = ("degraded_imu_gyro", "degraded_imu_acc", "imu_severity")
        absent = sorted(set(needed) - set(z.files))
        if absent:
            raise KeyError(f"Missing previously frozen preprocessing arrays: {absent}")
        d_g = np.asarray(z["degraded_imu_gyro"], float)
        d_a = np.asarray(z["degraded_imu_acc"], float)
        severity = np.asarray(z["imu_severity"], float)

    n = len(d_g)
    if n <= WINDOW or d_a.shape != (n,3) or severity.shape != (n,):
        raise ValueError(f"Invalid saved IMU array lengths, n={n}")

    loader = IMULoader(RAW_ROOT / date / args.sequence)
    if len(loader) < n:
        raise ValueError(f"Insufficient clean OXTS/IMU frames {len(loader)} < {n}")
    clean_acc = np.stack([
        np.asarray(loader[i]["acceleration"], float) for i in range(n)
    ])
    clean_gyro = np.stack([
        np.asarray(loader[i]["angular_velocity"], float) for i in range(n)
    ])
    gyro_err, acc_err = injected_imu_errors(
        clean_gyro, d_g, clean_acc, d_a
    )

    # Injection should be exactly zero at non-degraded frames. This also
    # checks that the frozen processed IMU and raw OXTS indices align.
    normal = (severity == 0.0)
    if normal.any():
        if np.max(gyro_err[normal]) > 1e-9 or np.max(acc_err[normal]) > 1e-9:
            raise ValueError(
                "Nonzero injected IMU delta on severity=0 frames; "
                "check raw-data integrity/frame ordering before interpreting results"
            )

    print("=" * 100)
    print("READ-ONLY KNOWN INJECTED IMU ERROR DIAGNOSTIC")
    print(f"sequence={args.sequence}, n={n}, full_window_valid={n - WINDOW + 1}")
    print("Stage-7 IMU FGO uses GYROSCOPE, not acceleration, in orientation factor.")
    print("Factor(i,i+1) samples gyro[i] but weights r[i+1], U[i+1].")
    print("Perturbation norm vs clean OXTS is NOT independent physical IMU ground truth.")
    print("No files will be modified.")
    for mode, indices in (
        ("same-frame", np.arange(WINDOW - 1, n)),
        ("factor-aligned", np.arange(WINDOW - 2, n - 1)),
    ):
        print(f"\nKnown injected delta / severity statistics [{mode}]")
        print("  gyro delta mean rad/s=", f"{gyro_err[indices].mean():.6f}")
        print("  acc  delta mean m/s² =", f"{acc_err[indices].mean():.6f}")
        print("  severity vs gyro delta:", association(severity[indices], gyro_err[indices]))
        print("  severity vs acc delta :", association(severity[indices], acc_err[indices]))

    for source in SENSORS:
        folder = out / "predictions" / source
        rpath = folder / "imu_reliability.txt"
        upath = folder / "imu_uncertainty.txt"
        if not rpath.is_file() or not upath.is_file():
            raise FileNotFoundError(f"Missing frozen prediction: {rpath} or {upath}")
        r = np.loadtxt(rpath, dtype=float).reshape(-1)
        u = np.loadtxt(upath, dtype=float).reshape(-1)
        if len(r) != n or len(u) != n or not np.isfinite(r).all() or not np.isfinite(u).all():
            raise ValueError(f"Invalid frozen predictions for {source}")
        print(f"\n[{source}]")
        for label, offset in (("same-frame", 0), ("factor-aligned", 1)):
            actual_g, impairment, unc = aligned_series(gyro_err, r, u, offset)
            actual_a, _, _ = aligned_series(acc_err, r, u, offset)
            print(f"  {label:<14} injected gyro vs 1-r: {association(actual_g, impairment)}")
            print(f"  {label:<14} injected gyro vs U  : {association(actual_g, unc)}")
            print(f"  {label:<14} injected acc  vs 1-r: {association(actual_a, impairment)}")
            print(f"  {label:<14} injected acc  vs U  : {association(actual_a, unc)}")
    print("\nIMU INJECTED-ERROR AUDIT COMPLETE (read-only)")


if __name__ == "__main__":
    main()
