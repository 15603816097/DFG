from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import FourSensorFactorConfig
from src.factor_graph.sensorwise_reliability_graph import (
    ReliabilitySource,
    run_sensorwise_reliability_graph,
)

# Reuse the already validated direct-sigma LiDAR graph implementation.
from experiments.factor_graph.run_lidar_calibrated_dynamic_fg import (
    run_dynamic_graph,
)

DEGRADED_DIR = ROOT / "results/degraded_four_sensor_measurements"
PHYSICAL_LIDAR = (
    ROOT / "results/lidar_physical_oracle_benchmark"
    / "physical_lidar_factor_data.npz"
)
V4_DIR = ROOT / "results/sensor_specific_reliability_v4"
LIDAR_MAPPING = (
    ROOT / "results/lidar_covariance_calibration_v1"
    / "mapping/lidar_dynamic_mapping.npz"
)
GT_PATH = ROOT / "results/ground_truth/trajectory.txt"
OUT = ROOT / "results/stage2_v4_factor_graph_ablation"

SIGMA_T0 = 0.15
SIGMA_R0 = 0.15 * (0.03 / 0.35)
LAMBDA = 1.0

CASES = [
    ("F0", "All Fixed", False, False, False, False),
    ("F1", "GPS V4", True, False, False, False),
    ("F2", "IMU V4", False, True, False, False),
    ("F3", "Camera V4", False, False, True, False),
    ("F4", "GPS + IMU V4", True, True, False, False),
    ("F5", "GPS + Camera V4", True, False, True, False),
    ("F6", "IMU + Camera V4", False, True, True, False),
    ("F7", "GPS + IMU + Camera V4; LiDAR fixed", True, True, True, False),
    ("F8", "GPS + IMU + Camera V4; LiDAR dynamic lambda=1", True, True, True, True),
]


def load_physical_measurements():
    measurements = load_degraded_four_sensor_measurements(DEGRADED_DIR)
    p = np.load(PHYSICAL_LIDAR, allow_pickle=False)

    between = np.asarray(p["body_between"], dtype=np.float64)
    valid = np.all(np.isfinite(between), axis=(1, 2))
    if "converged" in p.files:
        c = np.asarray(p["converged"]).reshape(-1).astype(bool)
        if len(c) == len(valid):
            valid &= c

    # The measurement object used in the project is mutable in the existing
    # experiment chain. Keep the replacement local to this process.
    measurements.lidar_between = between
    measurements.lidar_valid = valid
    return measurements


def load_v4_prediction(sensor, n):
    path = V4_DIR / f"{sensor}_predictive_prior_target_aligned.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    x = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if len(x) != n:
        raise ValueError(f"{sensor}: prediction length {len(x)} != frames {n}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{sensor}: non-finite prediction")
    return np.clip(x, 0.0, 1.0)


def make_prediction_dir(case_dir, n, gps_on, imu_on, camera_on):
    pred = case_dir / "predictive_inputs"
    pred.mkdir(parents=True, exist_ok=True)

    for sensor, enabled in (
        ("gps", gps_on),
        ("imu", imu_on),
        ("camera", camera_on),
    ):
        values = (
            load_v4_prediction(sensor, n)
            if enabled
            else np.ones(n, dtype=np.float64)
        )
        np.savetxt(
            pred / f"{sensor}_predictive_prior_target_aligned.txt",
            values,
            fmt="%.8f",
        )

    # sensorwise graph may look for LiDAR prediction depending on implementation.
    # F0-F7 always keep LiDAR fixed, so a neutral array is sufficient.
    np.savetxt(
        pred / "lidar_predictive_prior_target_aligned.txt",
        np.ones(n, dtype=np.float64),
        fmt="%.8f",
    )
    return pred


def evaluate_trajectory(trajectory):
    gt = np.loadtxt(GT_PATH, dtype=np.float64)
    est = np.asarray(trajectory, dtype=np.float64)
    n = min(len(gt), len(est))
    gt = gt[:n, :3]
    est = est[:n, :3]

    e = est - gt
    e3 = np.linalg.norm(e, axis=1)
    e2 = np.linalg.norm(e[:, :2], axis=1)

    return {
        "frames": int(n),
        "ATE3D": float(np.sqrt(np.mean(e3 ** 2))),
        "ATE2D": float(np.sqrt(np.mean(e2 ** 2))),
        "Mean3D": float(np.mean(e3)),
        "Max3D": float(np.max(e3)),
        "ZRMSE": float(np.sqrt(np.mean(e[:, 2] ** 2))),
    }


def fixed_config():
    cfg = FourSensorFactorConfig()
    return replace(
        cfg,
        lidar_translation_sigma=SIGMA_T0,
        lidar_rotation_sigma=SIGMA_R0,
    )


def source(enabled):
    return ReliabilitySource(mode="predictive" if enabled else "fixed")


def run_fixed_lidar_case(
    measurements,
    case_dir,
    pred_dir,
    gps_on,
    imu_on,
    camera_on,
):
    sources = {
        "gps": source(gps_on),
        "imu": source(imu_on),
        "lidar": ReliabilitySource(mode="fixed"),
        "camera": source(camera_on),
    }

    trajectory, poses, counts = run_sensorwise_reliability_graph(
        measurements=measurements,
        output_dir=case_dir,
        sources=sources,
        predictive_dir=pred_dir,
        config=fixed_config(),
    )
    return trajectory, poses, counts


def build_lambda1_mapping(case_dir):
    src = np.load(LIDAR_MAPPING, allow_pickle=False)

    if "risk_score_pair" not in src.files:
        raise KeyError(
            f"{LIDAR_MAPPING} has no risk_score_pair; keys={src.files}"
        )

    risk = np.asarray(src["risk_score_pair"], dtype=np.float64).reshape(-1)
    risk = np.clip(risk, 0.0, 1.0)
    scale = 1.0 + LAMBDA * risk ** 2

    sigma_t = SIGMA_T0 * scale
    sigma_r = SIGMA_R0 * scale

    path = case_dir / "lidar_lambda1_mapping.npz"
    np.savez_compressed(
        path,
        risk_score_pair=risk,
        translation_sigma_pair=sigma_t,
        rotation_sigma_pair=sigma_r,
        sigma_t0=np.asarray([SIGMA_T0]),
        sigma_r0=np.asarray([SIGMA_R0]),
        lambda_value=np.asarray([LAMBDA]),
    )

    print(
        "F8 LiDAR direct sigma t min/max/mean:",
        float(sigma_t.min()), float(sigma_t.max()), float(sigma_t.mean())
    )
    print(
        "F8 LiDAR direct sigma r min/max/mean:",
        float(sigma_r.min()), float(sigma_r.max()), float(sigma_r.mean())
    )
    return path


def run_dynamic_lidar_case(measurements, case_dir, pred_dir):
    mapping = build_lambda1_mapping(case_dir)

    # The validated runner uses module-level path variables. Override only for
    # this process so F8 consumes V4 G/I/C predictions and the explicit λ=1
    # direct covariance arrays, without modifying existing experiment files.
    import experiments.factor_graph.run_lidar_calibrated_dynamic_fg as dyn

    old_pred = dyn.PRED
    old_mapping = dyn.MAPPING
    old_out = dyn.OUT

    try:
        dyn.PRED = str(pred_dir)
        dyn.MAPPING = str(mapping)
        dyn.OUT = str(case_dir)

        mapping_data = np.load(mapping, allow_pickle=False)
        sigma_t_pair = np.asarray(
            mapping_data["translation_sigma_pair"],
            dtype=np.float64,
        ).reshape(-1)
        sigma_r_pair = np.asarray(
            mapping_data["rotation_sigma_pair"],
            dtype=np.float64,
        ).reshape(-1)

        expected_pairs = len(measurements.gps_local) - 1
        if len(sigma_t_pair) != expected_pairs:
            raise ValueError(
                f"F8 sigma_t length {len(sigma_t_pair)} != {expected_pairs}"
            )
        if len(sigma_r_pair) != expected_pairs:
            raise ValueError(
                f"F8 sigma_r length {len(sigma_r_pair)} != {expected_pairs}"
            )

        trajectory, poses, counts = dyn.run_dynamic_graph(
            measurements,
            sigma_t_pair,
            sigma_r_pair,
            str(case_dir),
        )
    finally:
        dyn.PRED = old_pred
        dyn.MAPPING = old_mapping
        dyn.OUT = old_out

    return trajectory, poses, counts


def save_metrics(case_dir, row):
    with (case_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        for k, v in row.items():
            f.write(f"{k}: {v}\n")


def main():
    for path in (DEGRADED_DIR, PHYSICAL_LIDAR, V4_DIR, LIDAR_MAPPING, GT_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)
    measurements = load_physical_measurements()
    n = len(measurements.gps_local)

    print("=" * 112)
    print("STAGE 2 — 0027 V4 FACTOR-GRAPH ABLATION")
    print("=" * 112)
    print("Frames:", n)
    print("LiDAR fixed sigma t/r:", SIGMA_T0, SIGMA_R0)
    print("LiDAR dynamic lambda:", LAMBDA)
    print("V4 prediction dir:", V4_DIR)
    print("GT:", GT_PATH)
    print("No training. No sigma sweep. No lambda sweep.")

    rows = []

    for code, name, gps_on, imu_on, camera_on, lidar_dynamic in CASES:
        print()
        print("=" * 112)
        print(code, name)
        print("=" * 112)

        case_dir = OUT / code
        case_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = make_prediction_dir(
            case_dir, n, gps_on, imu_on, camera_on
        )

        if lidar_dynamic:
            trajectory, poses, counts = run_dynamic_lidar_case(
                measurements, case_dir, pred_dir
            )
        else:
            trajectory, poses, counts = run_fixed_lidar_case(
                measurements,
                case_dir,
                pred_dir,
                gps_on,
                imu_on,
                camera_on,
            )

        metrics = evaluate_trajectory(trajectory)
        row = {
            "case": code,
            "name": name,
            "gps_v4": gps_on,
            "imu_v4": imu_on,
            "camera_v4": camera_on,
            "lidar_dynamic": lidar_dynamic,
            **metrics,
            "GPS_factors": int(counts.get("gps", 0)),
            "IMU_factors": int(counts.get("imu", 0)),
            "LiDAR_factors": int(counts.get("lidar", 0)),
            "Camera_factors": int(counts.get("camera", 0)),
        }
        rows.append(row)
        save_metrics(case_dir, row)

        print(
            f"{code}: ATE3D={metrics['ATE3D']:.6f} | "
            f"ATE2D={metrics['ATE2D']:.6f} | "
            f"Mean3D={metrics['Mean3D']:.6f} | "
            f"Max3D={metrics['Max3D']:.6f} | "
            f"ZRMSE={metrics['ZRMSE']:.6f}"
        )

    csv_path = OUT / "stage2_v4_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_code = {r["case"]: r for r in rows}
    f0, f7, f8 = by_code["F0"], by_code["F7"], by_code["F8"]

    improve_f8_f0 = 100.0 * (f0["ATE3D"] - f8["ATE3D"]) / f0["ATE3D"]
    improve_f8_f7 = 100.0 * (f7["ATE3D"] - f8["ATE3D"]) / f7["ATE3D"]

    ranked = sorted(rows, key=lambda r: r["ATE3D"])

    summary = OUT / "stage2_v4_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("STAGE 2 — 0027 V4 FACTOR-GRAPH ABLATION\n")
        f.write("=" * 90 + "\n\n")
        for i, r in enumerate(ranked, 1):
            f.write(
                f"{i:02d}. {r['case']} {r['name']}: "
                f"ATE3D={r['ATE3D']:.6f}, "
                f"ATE2D={r['ATE2D']:.6f}, "
                f"ZRMSE={r['ZRMSE']:.6f}\n"
            )
        f.write("\n")
        f.write(f"F8 improvement vs F0: {improve_f8_f0:+.2f}%\n")
        f.write(f"F8 improvement vs F7: {improve_f8_f7:+.2f}%\n")
        f.write(
            "NOTE: 0027 remains development/calibration evidence, "
            "not held-out cross-sequence test evidence.\n"
        )

    print()
    print("=" * 112)
    print("FINAL RANKING")
    print("=" * 112)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:02d}. {r['case']} | ATE3D={r['ATE3D']:.6f} | "
            f"ATE2D={r['ATE2D']:.6f} | ZRMSE={r['ZRMSE']:.6f} | {r['name']}"
        )

    print()
    print(f"F8 improvement vs F0: {improve_f8_f0:+.2f}%")
    print(f"F8 improvement vs F7: {improve_f8_f7:+.2f}%")
    print("CSV:", csv_path)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
