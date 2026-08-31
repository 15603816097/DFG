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
LIDAR_MAP = ROOT / "results" / "lidar_covariance_calibration_v1" / "mapping" / "lidar_dynamic_mapping.npz"
GT = ROOT / "results" / "ground_truth" / "trajectory.txt"

OUT = ROOT / "results" / "stage5_v5_gru_factor_graph_ablation"

# Frozen nominal anchors already validated on sequence 0027.
GPS_FIXED = 5.0
IMU_R0 = 0.03
CAM_T0 = 0.45
CAM_R0 = 0.04
LIDAR_T0 = 0.15
LIDAR_R0 = 0.012857142857142857


def load_measurements():
    base = load_degraded_four_sensor_measurements(BASE)
    m = SimpleNamespace(**vars(base))

    phys = np.load(PHYS, allow_pickle=False)
    m.lidar_between = np.asarray(phys["body_between"], dtype=float)
    m.lidar_quality = np.asarray(phys["quality"], dtype=float)
    m.lidar_valid = np.asarray(phys["converged"], dtype=bool)

    return m


def load_v5(sensor: str, n: int):
    r_path = V5 / f"{sensor}_reliability_target_aligned.txt"
    u_path = V5 / f"{sensor}_uncertainty_target_aligned.txt"

    if not r_path.exists():
        raise FileNotFoundError(r_path)
    if not u_path.exists():
        raise FileNotFoundError(u_path)

    r = np.loadtxt(r_path, dtype=float).reshape(-1)
    u = np.loadtxt(u_path, dtype=float).reshape(-1)

    if len(r) < n or len(u) < n:
        raise ValueError(
            f"{sensor}: prediction length mismatch, r={len(r)}, u={len(u)}, n={n}"
        )

    return np.clip(r[:n], 0.0, 1.0), np.clip(u[:n], 0.0, 1.0)


def load_lidar_risk(n_pairs: int):
    d = np.load(LIDAR_MAP, allow_pickle=False)
    risk = np.asarray(d["risk_score_pair"], dtype=float).reshape(-1)

    if len(risk) != n_pairs:
        raise ValueError(
            f"LiDAR risk length mismatch: {len(risk)} vs {n_pairs}"
        )

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
        "ZRMSE": float(np.sqrt(np.mean(err[:, 2]**2))),
    }


def direct_confidence(u: float) -> float:
    return 1.0


def uncertainty_confidence(u: float) -> float:
    return float(np.clip(1.0 - u, 0.0, 1.0))


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
    # Same frozen λ=1 mapping family as prior calibrated LiDAR experiments:
    # sigma = sigma0 * (1 + confidence * risk^2)
    scale = 1.0 + confidence * (float(risk) ** 2)
    return LIDAR_R0 * scale, LIDAR_T0 * scale


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

    gps_r, gps_u = load_v5("gps", n)
    imu_r, imu_u = load_v5("imu", n)
    cam_r, cam_u = load_v5("camera", n)
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

    for i in range(n):
        # ----------------------------------------------------
        # GPS
        # ----------------------------------------------------
        if gps_mode == "fixed":
            gps_sigma = GPS_FIXED
        elif gps_mode == "direct":
            gps_sigma = gps_dynamic_sigma(
                cfg,
                gps_r[i],
                direct_confidence(gps_u[i]),
            )
        elif gps_mode == "uncertainty":
            gps_sigma = gps_dynamic_sigma(
                cfg,
                gps_r[i],
                uncertainty_confidence(gps_u[i]),
            )
        else:
            raise ValueError(gps_mode)

        stats["gps"].append(gps_sigma)

        gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [1e6, 1e6, 1e6]
                + [gps_sigma, gps_sigma, gps_sigma],
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
        graph.add(
            gtsam.PriorFactorPose3(
                i,
                gps_pose,
                gps_noise,
            )
        )

        if i >= n - 1:
            continue

        dt = float(times[i + 1] - times[i])
        if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            dt = 0.1

        # ----------------------------------------------------
        # IMU
        # ----------------------------------------------------
        if imu_mode == "fixed":
            imu_sigma = IMU_R0
        elif imu_mode == "direct":
            imu_sigma = imu_dynamic_sigma(
                cfg,
                imu_r[i + 1],
                direct_confidence(imu_u[i + 1]),
            )
        elif imu_mode == "uncertainty":
            imu_sigma = imu_dynamic_sigma(
                cfg,
                imu_r[i + 1],
                uncertainty_confidence(imu_u[i + 1]),
            )
        else:
            raise ValueError(imu_mode)

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
                _imu_between_pose(
                    gtsam,
                    measurements.imu_gyro[i],
                    dt,
                ),
                imu_noise,
            )
        )

        # ----------------------------------------------------
        # LiDAR
        # ----------------------------------------------------
        if bool(measurements.lidar_valid[i]):
            if lidar_mode == "fixed":
                lidar_r_sigma = LIDAR_R0
                lidar_t_sigma = LIDAR_T0

            elif lidar_mode == "direct":
                lidar_r_sigma, lidar_t_sigma = lidar_dynamic_sigmas(
                    lidar_risk[i],
                    1.0,
                )

            elif lidar_mode == "uncertainty":
                # Dedicated LiDAR predictor currently has no evidential head.
                # Use risk ambiguity as conservative confidence proxy ONLY for U8.
                # risk near 0.5 => low confidence; risk near 0/1 => high confidence.
                lidar_conf = float(
                    np.clip(
                        2.0 * abs(lidar_risk[i] - 0.5),
                        0.0,
                        1.0,
                    )
                )
                lidar_r_sigma, lidar_t_sigma = lidar_dynamic_sigmas(
                    lidar_risk[i],
                    lidar_conf,
                )

            else:
                raise ValueError(lidar_mode)

            stats["lidar_t"].append(lidar_t_sigma)
            stats["lidar_r"].append(lidar_r_sigma)

            lidar_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [lidar_r_sigma] * 3
                    + [lidar_t_sigma] * 3,
                    dtype=float,
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    _pose3_from_matrix(
                        gtsam,
                        measurements.lidar_between[i],
                    ),
                    lidar_noise,
                )
            )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------
        if bool(measurements.camera_valid[i]):
            if camera_mode == "fixed":
                camera_r_sigma = CAM_R0
                camera_t_sigma = CAM_T0

            elif camera_mode == "direct":
                camera_r_sigma, camera_t_sigma = camera_dynamic_sigmas(
                    cfg,
                    cam_r[i + 1],
                    direct_confidence(cam_u[i + 1]),
                )

            elif camera_mode == "uncertainty":
                camera_r_sigma, camera_t_sigma = camera_dynamic_sigmas(
                    cfg,
                    cam_r[i + 1],
                    uncertainty_confidence(cam_u[i + 1]),
                )

            else:
                raise ValueError(camera_mode)

            stats["camera_t"].append(camera_t_sigma)
            stats["camera_r"].append(camera_r_sigma)

            camera_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [camera_r_sigma] * 3
                    + [camera_t_sigma] * 3,
                    dtype=float,
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    _pose3_from_matrix(
                        gtsam,
                        measurements.camera_between[i],
                    ),
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
        if len(arr) > 0:
            print(
                f"{key:10s} sigma min/max/mean "
                f"{arr.min():.6f}/"
                f"{arr.max():.6f}/"
                f"{arr.mean():.6f}"
            )

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-7)

    result = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial,
        params,
    ).optimize()

    poses = np.stack(
        [
            _pose3_to_matrix(
                result.atPose3(i)
            )
            for i in range(n)
        ],
        axis=0,
    )

    trajectory = poses[:, :3, 3]

    case_dir = OUT / name
    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        case_dir / "poses.npy",
        poses,
    )
    np.savetxt(
        case_dir / "trajectory.txt",
        trajectory,
        fmt="%.9f",
    )

    metrics = evaluate(trajectory)

    print("Metrics:", metrics)

    return metrics


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    measurements = load_measurements()

    cases = [
        # U0: all nominal fixed covariance
        (
            "U0_all_fixed",
            dict(
                gps_mode="fixed",
                imu_mode="fixed",
                camera_mode="fixed",
                lidar_mode="fixed",
            ),
        ),

        # GPS
        (
            "U1_gps_v5_direct",
            dict(
                gps_mode="direct",
                imu_mode="fixed",
                camera_mode="fixed",
                lidar_mode="fixed",
            ),
        ),
        (
            "U2_gps_v5_uncertainty",
            dict(
                gps_mode="uncertainty",
                imu_mode="fixed",
                camera_mode="fixed",
                lidar_mode="fixed",
            ),
        ),

        # IMU
        (
            "U3_imu_v5_direct",
            dict(
                gps_mode="fixed",
                imu_mode="direct",
                camera_mode="fixed",
                lidar_mode="fixed",
            ),
        ),
        (
            "U4_imu_v5_uncertainty",
            dict(
                gps_mode="fixed",
                imu_mode="uncertainty",
                camera_mode="fixed",
                lidar_mode="fixed",
            ),
        ),

        # Camera
        (
            "U5_camera_v5_direct",
            dict(
                gps_mode="fixed",
                imu_mode="fixed",
                camera_mode="direct",
                lidar_mode="fixed",
            ),
        ),
        (
            "U6_camera_v5_uncertainty",
            dict(
                gps_mode="fixed",
                imu_mode="fixed",
                camera_mode="uncertainty",
                lidar_mode="fixed",
            ),
        ),

        # GPS + IMU + Camera, LiDAR fixed
        (
            "U7_gic_v5_uncertainty_lidar_fixed",
            dict(
                gps_mode="uncertainty",
                imu_mode="uncertainty",
                camera_mode="uncertainty",
                lidar_mode="fixed",
            ),
        ),

        # Four-sensor uncertainty-aware dynamic
        (
            "U8_all_v5_uncertainty_dynamic",
            dict(
                gps_mode="uncertainty",
                imu_mode="uncertainty",
                camera_mode="uncertainty",
                lidar_mode="uncertainty",
            ),
        ),
    ]

    rows = []

    for name, kwargs in cases:
        metrics = run_case(
            name=name,
            measurements=measurements,
            **kwargs,
        )

        rows.append(
            {
                "case": name,
                **kwargs,
                **metrics,
            }
        )

    csv_path = OUT / "stage5_v5_gru_ablation.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    ranking = sorted(
        rows,
        key=lambda r: r["ATE3D"],
    )

    print()
    print("=" * 100)
    print("FINAL RANKING")
    print("=" * 100)

    for i, row in enumerate(ranking, 1):
        print(
            f"{i:02d}. {row['case']}: "
            f"ATE3D={row['ATE3D']:.6f}, "
            f"ATE2D={row['ATE2D']:.6f}, "
            f"ZRMSE={row['ZRMSE']:.6f}"
        )

    baseline = next(
        r for r in rows
        if r["case"] == "U0_all_fixed"
    )

    print()
    print("Improvement vs U0:")
    for row in rows:
        improvement = (
            baseline["ATE3D"]
            - row["ATE3D"]
        ) / baseline["ATE3D"] * 100.0

        print(
            f"{row['case']:38s}: "
            f"{improvement:+.2f}%"
        )

    print()
    print("Key uncertainty deltas:")
    pairs = [
        ("GPS", "U1_gps_v5_direct", "U2_gps_v5_uncertainty"),
        ("IMU", "U3_imu_v5_direct", "U4_imu_v5_uncertainty"),
        ("Camera", "U5_camera_v5_direct", "U6_camera_v5_uncertainty"),
    ]

    lookup = {
        r["case"]: r
        for r in rows
    }

    for sensor, direct_name, unc_name in pairs:
        direct = lookup[direct_name]["ATE3D"]
        unc = lookup[unc_name]["ATE3D"]

        gain = (
            direct - unc
        ) / direct * 100.0

        print(
            f"{sensor:8s}: "
            f"direct={direct:.6f}, "
            f"uncertainty={unc:.6f}, "
            f"gain={gain:+.2f}%"
        )

    print()
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
