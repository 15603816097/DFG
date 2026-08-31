from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
from src.factor_graph.sensorwise_reliability_graph import (
    _timestamp_seconds,
    _pose3_from_matrix,
    _pose3_to_matrix,
    _imu_between_pose,
    _initial_trajectory,
    _gps_sigma,
    _camera_sigmas,
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


BASE = ROOT / "results" / "degraded_four_sensor_measurements"
PHYS = ROOT / "results" / "lidar_physical_oracle_benchmark" / "physical_lidar_factor_data.npz"
V5 = ROOT / "results" / "sensor_specific_reliability_v5_uncertainty"
V6 = ROOT / "results" / "v6_mamba_uncertainty"
LIDAR_MAP = ROOT / "results" / "lidar_covariance_calibration_v1" / "mapping" / "lidar_dynamic_mapping.npz"
GT = ROOT / "results" / "ground_truth" / "trajectory.txt"

OUT = ROOT / "results" / "stage6_v6_mamba_factor_graph_ablation"

# Frozen nominal anchors: identical to Stage 5.
GPS_FIXED = 5.0
IMU_R0 = 0.03
CAM_T0 = 0.45
CAM_R0 = 0.04
LIDAR_T0 = 0.15
LIDAR_R0 = 0.012857142857142857

SENSORS = ("gps", "imu", "camera")
SOURCES = ("v5", "v6")


def load_measurements():
    base = load_degraded_four_sensor_measurements(BASE)
    m = SimpleNamespace(**vars(base))

    phys = np.load(PHYS, allow_pickle=False)
    m.lidar_between = np.asarray(phys["body_between"], dtype=float)
    m.lidar_quality = np.asarray(phys["quality"], dtype=float)
    m.lidar_valid = np.asarray(phys["converged"], dtype=bool)
    return m


def _prediction_paths(source: str, sensor: str):
    if source == "v5":
        base = V5
    elif source == "v6":
        base = V6
    else:
        raise ValueError(f"Unknown prediction source: {source}")

    return (
        base / f"{sensor}_reliability_target_aligned.txt",
        base / f"{sensor}_uncertainty_target_aligned.txt",
    )


def load_prediction(source: str, sensor: str, n: int):
    r_path, u_path = _prediction_paths(source, sensor)

    if not r_path.exists():
        raise FileNotFoundError(r_path)
    if not u_path.exists():
        raise FileNotFoundError(u_path)

    r = np.loadtxt(r_path, dtype=float).reshape(-1)
    u = np.loadtxt(u_path, dtype=float).reshape(-1)

    if len(r) < n or len(u) < n:
        raise ValueError(
            f"{source}/{sensor}: prediction length mismatch, "
            f"r={len(r)}, u={len(u)}, n={n}"
        )

    r = np.clip(r[:n], 0.0, 1.0)
    u = np.clip(u[:n], 0.0, 1.0)

    if not np.all(np.isfinite(r)):
        raise ValueError(f"{source}/{sensor}: reliability contains non-finite values")
    if not np.all(np.isfinite(u)):
        raise ValueError(f"{source}/{sensor}: uncertainty contains non-finite values")

    return r, u


def load_lidar_risk(n_pairs: int):
    d = np.load(LIDAR_MAP, allow_pickle=False)
    risk = np.asarray(d["risk_score_pair"], dtype=float).reshape(-1)

    if len(risk) != n_pairs:
        raise ValueError(f"LiDAR risk length mismatch: {len(risk)} vs {n_pairs}")

    return np.clip(risk, 0.0, 1.0)


def evaluate(trajectory: np.ndarray):
    gt = np.loadtxt(GT, dtype=float)
    est = np.asarray(trajectory, dtype=float)

    if gt.ndim != 2 or est.ndim != 2:
        raise ValueError("Ground truth and estimate must be 2-D arrays.")

    if gt.shape[1] > 3:
        gt = gt[:, -3:]
    if est.shape[1] > 3:
        est = est[:, -3:]

    n = min(len(gt), len(est))
    err = est[:n, :3] - gt[:n, :3]

    d3 = np.linalg.norm(err, axis=1)
    d2 = np.linalg.norm(err[:, :2], axis=1)

    return {
        "ATE3D": float(np.sqrt(np.mean(d3**2))),
        "ATE2D": float(np.sqrt(np.mean(d2**2))),
        "Mean3D": float(np.mean(d3)),
        "Max3D": float(np.max(d3)),
        "ZRMSE": float(np.sqrt(np.mean(err[:, 2] ** 2))),
    }


def uncertainty_confidence(u: float) -> float:
    return float(np.clip(1.0 - float(u), 0.0, 1.0))


def gps_dynamic_sigma(cfg, r, confidence):
    predictive = _gps_sigma(cfg, "predictive", float(r))
    return GPS_FIXED + confidence * (predictive - GPS_FIXED)


def imu_dynamic_sigma(cfg, r, confidence):
    predictive = (
        cfg.imu_rot_sigma_min
        + (1.0 - float(r))
        * (cfg.imu_rot_sigma_max - cfg.imu_rot_sigma_min)
    )
    return IMU_R0 + confidence * (predictive - IMU_R0)


def camera_dynamic_sigmas(cfg, r, confidence):
    pred_r, pred_t = _camera_sigmas(cfg, "predictive", float(r))
    sigma_r = CAM_R0 + confidence * (pred_r - CAM_R0)
    sigma_t = CAM_T0 + confidence * (pred_t - CAM_T0)
    return sigma_r, sigma_t


def lidar_dynamic_sigmas(risk, confidence):
    # Frozen Stage-5 LiDAR mapping. It is intentionally unchanged so that
    # GRU-vs-Mamba comparisons modify only GPS/IMU/Camera prediction source.
    scale = 1.0 + confidence * (float(risk) ** 2)
    return LIDAR_R0 * scale, LIDAR_T0 * scale


def _sensor_mode(mode: str):
    """
    Supported:
      fixed
      v5_uncertainty
      v6_uncertainty
    """
    if mode == "fixed":
        return "fixed", None
    if mode == "v5_uncertainty":
        return "uncertainty", "v5"
    if mode == "v6_uncertainty":
        return "uncertainty", "v6"
    raise ValueError(f"Unsupported sensor mode: {mode}")


def run_case(
    name: str,
    measurements,
    gps_mode: str = "fixed",
    imu_mode: str = "fixed",
    camera_mode: str = "fixed",
    lidar_mode: str = "fixed",
):
    import gtsam

    n = len(measurements.gps_local)
    cfg = FourSensorFactorConfig()

    prediction_cache = {}
    for source in SOURCES:
        for sensor in SENSORS:
            prediction_cache[(source, sensor)] = load_prediction(source, sensor, n)

    lidar_risk = load_lidar_risk(n - 1)

    poses0 = _initial_trajectory(gtsam, measurements)
    times = _timestamp_seconds(measurements.timestamps)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array(
            [cfg.prior_rotation_sigma] * 3
            + [cfg.prior_translation_sigma] * 3,
            dtype=float,
        )
    )
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), prior_noise))

    for i, pose in enumerate(poses0):
        initial.insert(i, pose)

    stats = {
        "gps": [],
        "imu": [],
        "camera_t": [],
        "camera_r": [],
        "lidar_t": [],
        "lidar_r": [],
    }

    gps_kind, gps_source = _sensor_mode(gps_mode)
    imu_kind, imu_source = _sensor_mode(imu_mode)
    cam_kind, cam_source = _sensor_mode(camera_mode)

    for i in range(n):
        # GPS
        if gps_kind == "fixed":
            gps_sigma = GPS_FIXED
        else:
            gps_r, gps_u = prediction_cache[(gps_source, "gps")]
            gps_sigma = gps_dynamic_sigma(
                cfg, gps_r[i], uncertainty_confidence(gps_u[i])
            )

        stats["gps"].append(gps_sigma)

        gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [1e6, 1e6, 1e6] + [gps_sigma, gps_sigma, gps_sigma],
                dtype=float,
            )
        )

        gps_pose = gtsam.Pose3(
            gtsam.Rot3(),
            gtsam.Point3(
                float(measurements.gps_local[i, 0]),
                float(measurements.gps_local[i, 1]),
                float(measurements.gps_local[i, 2]),
            ),
        )
        graph.add(gtsam.PriorFactorPose3(i, gps_pose, gps_noise))

        if i >= n - 1:
            continue

        dt = float(times[i + 1] - times[i])
        if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            dt = 0.1

        # IMU
        if imu_kind == "fixed":
            imu_sigma = IMU_R0
        else:
            imu_r, imu_u = prediction_cache[(imu_source, "imu")]
            imu_sigma = imu_dynamic_sigma(
                cfg, imu_r[i + 1], uncertainty_confidence(imu_u[i + 1])
            )

        stats["imu"].append(imu_sigma)

        imu_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [imu_sigma, imu_sigma, imu_sigma]
                + [cfg.imu_translation_sigma] * 3,
                dtype=float,
            )
        )
        graph.add(
            gtsam.BetweenFactorPose3(
                i,
                i + 1,
                _imu_between_pose(gtsam, measurements.imu_gyro[i], dt),
                imu_noise,
            )
        )

        # LiDAR
        if bool(measurements.lidar_valid[i]):
            if lidar_mode == "fixed":
                lidar_r_sigma = LIDAR_R0
                lidar_t_sigma = LIDAR_T0
            elif lidar_mode == "uncertainty":
                # Same Stage-5 conservative ambiguity proxy.
                lidar_conf = float(
                    np.clip(2.0 * abs(lidar_risk[i] - 0.5), 0.0, 1.0)
                )
                lidar_r_sigma, lidar_t_sigma = lidar_dynamic_sigmas(
                    lidar_risk[i], lidar_conf
                )
            else:
                raise ValueError(f"Unsupported lidar_mode: {lidar_mode}")

            stats["lidar_t"].append(lidar_t_sigma)
            stats["lidar_r"].append(lidar_r_sigma)

            lidar_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [lidar_r_sigma] * 3 + [lidar_t_sigma] * 3,
                    dtype=float,
                )
            )
            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    _pose3_from_matrix(gtsam, measurements.lidar_between[i]),
                    lidar_noise,
                )
            )

        # Camera
        if bool(measurements.camera_valid[i]):
            if cam_kind == "fixed":
                camera_r_sigma = CAM_R0
                camera_t_sigma = CAM_T0
            else:
                cam_r, cam_u = prediction_cache[(cam_source, "camera")]
                camera_r_sigma, camera_t_sigma = camera_dynamic_sigmas(
                    cfg,
                    cam_r[i + 1],
                    uncertainty_confidence(cam_u[i + 1]),
                )

            stats["camera_t"].append(camera_t_sigma)
            stats["camera_r"].append(camera_r_sigma)

            camera_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [camera_r_sigma] * 3 + [camera_t_sigma] * 3,
                    dtype=float,
                )
            )
            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    _pose3_from_matrix(gtsam, measurements.camera_between[i]),
                    camera_noise,
                )
            )

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)
    print("Factors:", graph.size())
    print(
        f"Modes: GPS={gps_mode}, IMU={imu_mode}, "
        f"Camera={camera_mode}, LiDAR={lidar_mode}"
    )

    for key, values in stats.items():
        arr = np.asarray(values, dtype=float)
        if len(arr):
            print(
                f"{key:10s} sigma min/max/mean "
                f"{arr.min():.6f}/{arr.max():.6f}/{arr.mean():.6f}"
            )

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-7)

    result = gtsam.LevenbergMarquardtOptimizer(
        graph, initial, params
    ).optimize()

    poses = np.stack(
        [_pose3_to_matrix(result.atPose3(i)) for i in range(n)],
        axis=0,
    )
    trajectory = poses[:, :3, 3]

    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "poses.npy", poses)
    np.savetxt(case_dir / "trajectory.txt", trajectory, fmt="%.9f")

    metrics = evaluate(trajectory)
    print("Metrics:", metrics)
    return metrics


def build_cases():
    """
    G0: exact all-fixed control.
    G1: Stage-5 GRU U8 reproduction.
    G2: official-Mamba replacement for GPS/IMU/Camera; same LiDAR policy.
    G3: hybrid chosen strictly from frozen predictor-level validation:
        GPS=Mamba, IMU=GRU, Camera=Mamba. No FG result is used to choose it.
    G4-G6: isolate each Mamba sensor against an otherwise fixed graph.
    """
    return [
        (
            "G0_all_fixed",
            dict(
                gps_mode="fixed",
                imu_mode="fixed",
                camera_mode="fixed",
                lidar_mode="fixed",
            ),
        ),
        (
            "G1_v5_gru_all_uncertainty",
            dict(
                gps_mode="v5_uncertainty",
                imu_mode="v5_uncertainty",
                camera_mode="v5_uncertainty",
                lidar_mode="uncertainty",
            ),
        ),
        (
            "G2_v6_mamba_all_uncertainty",
            dict(
                gps_mode="v6_uncertainty",
                imu_mode="v6_uncertainty",
                camera_mode="v6_uncertainty",
                lidar_mode="uncertainty",
            ),
        ),
        (
            "G3_hybrid_gps_mamba_imu_gru_camera_mamba",
            dict(
                gps_mode="v6_uncertainty",
                imu_mode="v5_uncertainty",
                camera_mode="v6_uncertainty",
                lidar_mode="uncertainty",
            ),
        ),
        (
            "G4_gps_mamba_only",
            dict(
                gps_mode="v6_uncertainty",
                imu_mode="fixed",
                camera_mode="fixed",
                lidar_mode="fixed",
            ),
        ),
        (
            "G5_imu_mamba_only",
            dict(
                gps_mode="fixed",
                imu_mode="v6_uncertainty",
                camera_mode="fixed",
                lidar_mode="fixed",
            ),
        ),
        (
            "G6_camera_mamba_only",
            dict(
                gps_mode="fixed",
                imu_mode="fixed",
                camera_mode="v6_uncertainty",
                lidar_mode="fixed",
            ),
        ),
    ]


def preflight():
    required = [BASE, PHYS, V5, V6, LIDAR_MAP, GT]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 6 required inputs are missing:\n  - " + "\n  - ".join(missing)
        )

    # Validate all prediction files before starting expensive GTSAM optimization.
    base = load_degraded_four_sensor_measurements(BASE)
    n = len(base.gps_local)
    for source in SOURCES:
        for sensor in SENSORS:
            r, u = load_prediction(source, sensor, n)
            print(
                f"[preflight] {source}/{sensor}: n={len(r)}, "
                f"r_mean={r.mean():.6f}, u_mean={u.mean():.6f}"
            )

    risk = load_lidar_risk(n - 1)
    print(f"[preflight] lidar risk: n={len(risk)}, mean={risk.mean():.6f}")
    print("[preflight] PASS")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preflight()
    measurements = load_measurements()

    rows = []
    for name, kwargs in build_cases():
        metrics = run_case(
            name=name,
            measurements=measurements,
            **kwargs,
        )
        rows.append({"case": name, **kwargs, **metrics})

    baseline = next(r for r in rows if r["case"] == "G0_all_fixed")
    gru = next(r for r in rows if r["case"] == "G1_v5_gru_all_uncertainty")
    mamba = next(r for r in rows if r["case"] == "G2_v6_mamba_all_uncertainty")

    for row in rows:
        row["ImprovementVsFixedPct"] = (
            (baseline["ATE3D"] - row["ATE3D"]) / baseline["ATE3D"] * 100.0
        )

    csv_path = OUT / "stage6_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ranking = sorted(rows, key=lambda r: r["ATE3D"])

    summary_path = OUT / "stage6_summary.txt"
    lines = []
    lines.append("Stage 6: V5-GRU vs V6 Official Mamba Factor-Graph Comparison")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Ranking:")
    for i, row in enumerate(ranking, 1):
        lines.append(
            f"{i:02d}. {row['case']}: "
            f"ATE3D={row['ATE3D']:.6f}, "
            f"ATE2D={row['ATE2D']:.6f}, "
            f"ZRMSE={row['ZRMSE']:.6f}, "
            f"vs_fixed={row['ImprovementVsFixedPct']:+.2f}%"
        )

    encoder_gain = (gru["ATE3D"] - mamba["ATE3D"]) / gru["ATE3D"] * 100.0
    lines += [
        "",
        "Primary encoder comparison:",
        f"G1 V5-GRU ATE3D   = {gru['ATE3D']:.6f}",
        f"G2 V6-Mamba ATE3D = {mamba['ATE3D']:.6f}",
        f"Mamba gain vs GRU = {encoder_gain:+.2f}%",
        "",
        "Sanity targets from the previously validated Stage 5 run:",
        "G0 should reproduce approximately 2.456340 m.",
        "G1 should reproduce approximately 2.371267 m.",
        "Small floating-point differences are acceptable; large differences require investigation.",
        "",
        "Important: LiDAR policy is frozen between G1/G2/G3. "
        "Therefore the primary G1-vs-G2 change is the GPS/IMU/Camera predictor source.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 100)
    print("FINAL STAGE-6 RANKING")
    print("=" * 100)
    for line in lines[4:4 + len(ranking)]:
        print(line)

    print()
    print(f"G1 GRU  ATE3D: {gru['ATE3D']:.6f}")
    print(f"G2 Mamba ATE3D: {mamba['ATE3D']:.6f}")
    print(f"Mamba gain vs GRU: {encoder_gain:+.2f}%")
    print("CSV:", csv_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
