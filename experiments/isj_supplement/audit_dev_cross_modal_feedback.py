from __future__ import annotations

"""READ-ONLY 0027 diagnostics for GPS/IMU-vs-LiDAR feedback and Beta update.

Never edits models/checkpoints, reference trajectories, labels, or Stage-7
results. Does not run FGO. q is computed exclusively from *degraded*
GPS/IMU plus *physical* LiDAR odometry, not from clean GPS/GT/oracle labels.
V3 targets and injected severities appear ONLY AFTER the feedback
computation in diagnostic score printing; they are NOT used to tune q.

0027 predictions were generated from a model trained on this very sequence
and may include in-sample predictions. These statistics are descriptive,
not held-out evidence. V6 checkpoint lineage is not yet independently proven.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.cross_modal_consistency import (
    CrossModalSettings, measurement_consistency,
)
from src.reliability.evidence_feedback_calibration import (
    CalibrationSettings, beta_soft_evidence_update,
)

SENSORS = ("gps", "imu")
MODELS = {
    "GRU": ROOT / "results" / "sensor_specific_reliability_v5_uncertainty",
    "MAMBA": ROOT / "results" / "v6_mamba_uncertainty",
}
NOMINAL = CrossModalSettings()
CAL = CalibrationSettings()  # declared k=2, warmup=63; NO tuning on 0016.


def pick(z, *names):
    for name in names:
        if name in z.files:
            return np.asarray(z[name])
    raise KeyError(f"Missing {names}; got {z.files}")


def describe(name, values):
    v = np.asarray(values, dtype=float).reshape(-1)
    if len(v) == 0 or not np.isfinite(v).all():
        return f"{name}: no valid finite observations"
    return (
        f"{name:<18} n={len(v):5d} mean={v.mean():.6f} "
        f"p05={np.percentile(v,5):.6f} "
        f"p50={np.median(v):.6f} p95={np.percentile(v,95):.6f}"
    )


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.shape != b.shape or len(a) < 3 or
            not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.std(a) < 1e-12 or np.std(b) < 1e-12):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    base = ROOT / "results" / "degraded_four_sensor_measurements"
    lidar_path = ROOT / "results" / "lidar_physical_oracle_benchmark" / "physical_lidar_factor_data.npz"
    labels_path = ROOT / "results" / "predictive_factor_reliability_v3_same_source" / "factor_reliability_training_data.npz"
    sensor_path = base / "degraded_sensor_data.npz"
    for path in (sensor_path, lidar_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(sensor_path, allow_pickle=False) as z:
        gps = np.asarray(z["degraded_gps_local"], dtype=float)
        gyro = np.asarray(z["degraded_imu_gyro"], dtype=float)
        ts = np.asarray(z["timestamps_seconds"], dtype=float)
        sev = {}
        for sensor in SENSORS:
            key = f"{sensor}_severity"
            if key in z.files:
                sev[sensor] = np.asarray(z[key], dtype=float).reshape(-1)
    n = len(gps)
    with np.load(lidar_path, allow_pickle=False) as z:
        between = np.asarray(pick(z, "body_between", "between_measurements"), float)
        valid = np.asarray(pick(z, "converged", "valid"), bool).reshape(-1)
        quality = np.asarray(z["quality"], float).reshape(-1) if "quality" in z.files else np.ones(len(valid), dtype=float)
        if "quality" not in z.files:
            print("WARNING: Physical LiDAR quality absent, using validity alone")

    if between.shape[0] < n-1 or len(valid) < n-1 or len(quality) < n-1:
        raise ValueError("Physical LiDAR odometry is shorter than GPS/IMU stream")
    feedback = measurement_consistency(
        gps_local=gps, imu_gyro=gyro, timestamps_seconds=ts,
        lidar_between=between[:n-1], lidar_valid=valid[:n-1],
        lidar_quality=quality[:n-1], settings=NOMINAL,
    )
    with np.load(labels_path, allow_pickle=False) as labels:
        reference = {
            sensor: np.asarray(
                labels[f"{sensor}_current_factor_reliability"], dtype=float
            ).reshape(-1)
            for sensor in SENSORS
        }
    if any(len(reference[s]) != n for s in SENSORS):
        raise ValueError("V3 diagnostic label length mismatch")
    if any(s in sev and len(sev[s]) != n for s in SENSORS):
        raise ValueError("Severity label length mismatch")

    print("="*104)
    print("DEVELOPMENT-ONLY READ-ONLY GPS/IMU CROSS-MODAL FEEDBACK DIAGNOSTIC")
    print(f"0027 frames={n}, valid model output starts at frame={CAL.expected_warmup}")
    print(f"GPS window={NOMINAL.gps_window}, GPS scalar chi2(1)95={NOMINAL.gps_chi2_95:.4f}")
    print(f"IMU chi2(3)95={NOMINAL.imu_chi2_95:.4f}, beta pseudo_count={CAL.pseudo_count}")
    print("NO ground truth / clean GPS / oracle used in feedback construction")
    print("WARNING: 0027 is development data; predictions are not independent test")
    print("WARNING: q is consistency score, NOT a posterior reliability probability")
    print("="*104)

    for sensor in SENSORS:
        q = feedback[f"{sensor}_quality"]
        mask = feedback[f"{sensor}_valid"].copy()
        mask[:CAL.expected_warmup] = False
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"{sensor}: no valid aligned cross-modal observations")
        print(f"\n--- {sensor.upper()} cross-modal consistency ---")
        print(f"feedback support={count}/{n-CAL.expected_warmup} "
              f"({count/max(1,n-CAL.expected_warmup):.2%})")
        print(describe("NIS", feedback[f"{sensor}_nis"][mask]))
        print(describe("feedback q", q[mask]))
        print(f"q<1 fraction={np.mean(q[mask]<1-1e-7):.4%}; "
              f"q<0.8 fraction={np.mean(q[mask]<.8):.4%}")
        if sensor in sev:
            print(f"Pearson feedback q vs injected severity: {corr(q[mask],sev[sensor][mask]):+.5f}")

        for name, folder in MODELS.items():
            rp = folder / f"{sensor}_reliability_target_aligned.txt"
            up = folder / f"{sensor}_uncertainty_target_aligned.txt"
            for p in (rp, up):
                if not p.is_file():
                    raise FileNotFoundError(p)
            r = np.asarray(np.loadtxt(rp), dtype=float).reshape(-1)
            u = np.asarray(np.loadtxt(up), dtype=float).reshape(-1)
            if len(r) != n or len(u) != n:
                raise ValueError(f"{name}/{sensor}: prediction length mismatch")
            calibrated, weights = beta_soft_evidence_update(
                r, u, q, valid_feedback=mask, settings=CAL,
            )
            t = reference[sensor]
            evalmask = np.arange(n) >= CAL.expected_warmup
            before = float(np.mean(np.abs(r[evalmask]-t[evalmask])))
            after = float(np.mean(np.abs(calibrated[evalmask]-t[evalmask])))
            print(
                f"{name:5s} r_MAE={before:.6f} -> r*_MAE={after:.6f} "
                f"(in-sample diagnostic, not GT-independent); "
                f"delta={after-before:+.6f} "
                f"mean_feedback_weight={weights[mask].mean():.6f} "
                f"mean_abs_dr={np.mean(np.abs(calibrated[mask]-r[mask])):.6f} "
                f"Pearson(U, |r-ref|)={corr(u[evalmask],np.abs(r[evalmask]-t[evalmask])):+.5f}"
            )
    print("\nAUDIT COMPLETE: no files written; NO FGO performed")


if __name__ == "__main__":
    main()
