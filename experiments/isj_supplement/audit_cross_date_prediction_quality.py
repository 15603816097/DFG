from __future__ import annotations

"""READ-ONLY audit of frozen cross-date reliability and evidential U.

Known synthetic inputs:
  - GPS: injected position displacement |degraded_gps - clean_gps| (meters).
  - GPS, IMU and Camera: injected degradation severity in [0,1].

Important: severity is a synthetic control parameter, NOT the original
ground-truth factor-reliability label, and U=2/(alpha+beta) is Beta
evidential lack-of-evidence, NOT necessarily measurement noise variance.
Correlations below are descriptive on an explored cross-date pilot:
do not interpret them as formal uncertainty calibration or blind generalization.
The GPS corruption reference shares the OXTS source with localization GT.
Only frames t>=63 have full frozen-encoder windows.
Never changes predictions, preprocessing, model weights or FGO outputs.
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
    RESULT_ROOT, date_for, validate_destination,
)

WINDOW = 64
SENSORS = ("gps", "imu", "camera")
ENCODERS = ("v5", "v6")


def pearson(x, y):
    a, b = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if a.shape != b.shape or len(a) < 3:
        raise ValueError("Correlation arrays must have equal length >=3")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correlation arrays must be finite")
    a, b = a - a.mean(), b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 1e-14 else math.nan


def tied_ranks(x):
    a = np.asarray(x, dtype=float).reshape(-1)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    k = 0
    while k < len(a):
        j = k + 1
        while j < len(a) and a[order[j]] == a[order[k]]:
            j += 1
        ranks[order[k:j]] = 0.5 * (k + j - 1) + 1
        k = j
    return ranks


def spearman(x, y):
    return pearson(tied_ranks(x), tied_ranks(y))


def beta_invalid(r, u):
    return np.mean(
        (u <= 0) | (u > 1) | (r < 0) | (r > 1) |
        (u > 2 * np.minimum(r, 1-r) + 1e-3)
    )


def _fmt(x):
    return "constant/NA" if not math.isfinite(x) else f"{x:+.4f}"


def load_raw(out: Path) -> tuple[dict, int]:
    path = out / "sensor_data.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "clean_gps_local", "degraded_gps_local",
            "gps_severity", "imu_severity", "camera_severity",
        }
        absent = sorted(required - set(data.files))
        if absent:
            raise KeyError(f"Missing saved synthetic-ground-truth keys: {absent}")
        arrays = {key: np.asarray(data[key], dtype=float) for key in required}
    clean, degraded = arrays["clean_gps_local"], arrays["degraded_gps_local"]
    if clean.shape != degraded.shape or clean.ndim != 2 or clean.shape[1] != 3:
        raise ValueError("Expected clean/degraded GPS arrays shape (N,3)")
    n = len(clean)
    if n <= WINDOW:
        raise ValueError(f"Too few frames to audit: {n}")
    for sensor in SENSORS:
        y = arrays[f"{sensor}_severity"]
        if y.shape != (n,) or not np.isfinite(y).all():
            raise ValueError(f"Invalid {sensor} severity shape: {y.shape}")
        if np.any((y < 0) | (y > 1)):
            raise ValueError("Synthetic severities must be within [0,1]")
    if not np.isfinite(clean).all() or not np.isfinite(degraded).all():
        raise ValueError("GPS coordinates contain NaN/Inf")
    arrays["gps_injected_error_m"] = np.linalg.norm(degraded-clean, axis=1)
    return arrays, n


def load_frozen(out: Path, source: str, sensor: str, n: int):
    folder = out / "predictions" / source
    paths = [folder / f"{sensor}_{key}.txt" for key in
             ("reliability", "uncertainty")]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(f"Frozen prediction missing: {paths}")
    r, u = [np.loadtxt(p, dtype=float).reshape(-1) for p in paths]
    if len(r) != n or len(u) != n or not np.isfinite(r).all() or not np.isfinite(u).all():
        raise ValueError(f"Invalid predictions: {source}/{sensor}")
    return r[WINDOW-1:], u[WINDOW-1:]


def summary_one(source, sensor, r, u, severity, error):
    print(
        f"\n[{source}/{sensor}] valid={len(r)} | "
        f"mean(r)={r.mean():.4f} mean(U)={u.mean():.4f} "
        f"Beta_invalid={100*beta_invalid(r,u):.2f}%"
    )
    # Reliability r is high for reliable sensors; use 1-r as
    # the signed, monotonic "predicted impairment" indicator.
    impairment = 1-r
    print(
        "  injected severity vs (1-r): "
        f"Pearson={_fmt(pearson(severity,impairment))}, "
        f"Spearman={_fmt(spearman(severity,impairment))}"
    )
    print(
        "  injected severity vs U    : "
        f"Pearson={_fmt(pearson(severity,u))}, "
        f"Spearman={_fmt(spearman(severity,u))}"
    )
    for label, lo, hi in (
        ("normal", 0, 0),
        ("mild", 0, 0.35),
        ("medium", 0.35, 0.7),
        ("severe", 0.7, 1.000001),
    ):
        if label == "normal":
            mask = severity == 0
        else:
            mask = (severity > lo) & (severity <= hi)
        if not np.any(mask):
            print(f"  severity {label:<6}: n=0")
            continue
        value = (
            f" injected_gps_error_m={error[mask].mean():.4f}"
            if error is not None else ""
        )
        print(
            f"  severity {label:<6}: n={mask.sum():3d} "
            f"r={r[mask].mean():.4f} U={u[mask].mean():.4f}{value}"
        )
    if error is not None:
        print(
            "  KNOWN GPS injected position error vs (1-r): "
            f"Pearson={_fmt(pearson(error,impairment))}, "
            f"Spearman={_fmt(spearman(error,impairment))}"
        )
        print(
            "  KNOWN GPS injected position error vs U:     "
            f"Pearson={_fmt(pearson(error,u))}, "
            f"Spearman={_fmt(spearman(error,u))}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    args = p.parse_args()
    date_for(args.sequence)
    out = validate_destination(args.sequence, RESULT_ROOT)
    data, n = load_raw(out)
    print("=" * 100)
    print("READ-ONLY FROZEN PREDICTION QUALITY AUDIT")
    print(f"sequence={args.sequence} frames={n}, full_window_frames={n-WINDOW+1}")
    print("GPS error = known synthetic injected displacement, NOT independent pose truth")
    print("Severity = synthetic control/proxy, NOT the training reliability label")
    print("U = Beta evidential lack-of-evidence, NOT inherently Gaussian noise sigma")
    print("Correlations are descriptive, NOT calibrated uncertainty scoring")
    for source in ENCODERS:
        for sensor in SENSORS:
            r, u = load_frozen(out, source, sensor, n)
            severity = data[f"{sensor}_severity"][WINDOW-1:]
            gps_err = (data["gps_injected_error_m"][WINDOW-1:]
                       if sensor == "gps" else None)
            summary_one(source, sensor, r, u, severity, gps_err)
    print("\nAUDIT COMPLETE (no files written)")


if __name__ == "__main__":
    main()
