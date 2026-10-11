from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from experiments.cross_sequence.stage7_config import *
from src.factor_graph.sensorwise_reliability_graph import (
    _pose3_from_matrix,
    _pose3_to_matrix,
    _imu_between_pose,
    _initial_trajectory,
    _gps_sigma,
    _camera_sigmas,
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


def _pick(z, *names):
    """
    Read the first matching field from an NPZ file.

    Stage-7 physical factor files may use either singular or plural
    naming depending on the generator version.
    """
    for name in names:
        if name in z.files:
            return np.asarray(z[name])

    raise KeyError(
        f"None of fields {names} found. "
        f"Available fields: {list(z.files)}"
    )


def load_measurements(seq):
    od = sequence_out(seq)

    sensor_path = od / "sensor_data.npz"
    lidar_path = od / "lidar_factor_data.npz"
    camera_path = od / "camera_factor_data.npz"

    for path in (sensor_path, lidar_path, camera_path):
        if not path.exists():
            raise FileNotFoundError(path)

    d = np.load(sensor_path, allow_pickle=False)
    l = np.load(lidar_path, allow_pickle=False)
    c = np.load(camera_path, allow_pickle=False)

    gps_local = np.asarray(d["degraded_gps_local"], dtype=float)
    imu_acc = np.asarray(d["degraded_imu_acc"], dtype=float)
    imu_gyro = np.asarray(d["degraded_imu_gyro"], dtype=float)
    timestamps = np.asarray(d["timestamps_seconds"], dtype=float)

    if "initial_oxts_rpy" not in d.files:
        raise KeyError(
            f"{seq}: sensor_data.npz is missing initial_oxts_rpy. "
            "Stage-7 requires the sequence's frozen OXTS initial attitude."
        )

    initial_oxts_rpy = np.asarray(
        d["initial_oxts_rpy"],
        dtype=float,
    ).reshape(-1)

    if initial_oxts_rpy.size != 3 or not np.isfinite(initial_oxts_rpy).all():
        raise ValueError(
            f"{seq}: invalid initial_oxts_rpy: {initial_oxts_rpy}"
        )

    lidar_between = _pick(
        l,
        "between_measurements",
        "between_measurement",
        "body_between",
        "between",
    ).astype(float)

    lidar_valid = _pick(
        l,
        "valid",
        "converged",
        "success",
    ).astype(bool)

    lidar_quality = _pick(
        l,
        "quality",
    ).astype(float)

    camera_between = _pick(
        c,
        "between_measurements",
        "between_measurement",
        "body_between",
        "between",
    ).astype(float)

    camera_valid = _pick(
        c,
        "valid",
        "converged",
        "success",
    ).astype(bool)

    camera_quality = _pick(
        c,
        "quality",
    ).astype(float)

    n = len(gps_local)

    if len(imu_acc) != n:
        raise ValueError(
            f"{seq}: imu_acc length mismatch: "
            f"{len(imu_acc)} != {n}"
        )

    if len(imu_gyro) != n:
        raise ValueError(
            f"{seq}: imu_gyro length mismatch: "
            f"{len(imu_gyro)} != {n}"
        )

    if len(timestamps) != n:
        raise ValueError(
            f"{seq}: timestamps length mismatch: "
            f"{len(timestamps)} != {n}"
        )

    # Between factors normally contain n-1 measurements.
    if len(lidar_between) < n - 1:
        raise ValueError(
            f"{seq}: LiDAR between-factor length too short: "
            f"{len(lidar_between)} < {n - 1}"
        )

    if len(lidar_valid) < n - 1:
        raise ValueError(
            f"{seq}: LiDAR valid length too short: "
            f"{len(lidar_valid)} < {n - 1}"
        )

    if len(camera_between) < n - 1:
        raise ValueError(
            f"{seq}: Camera between-factor length too short: "
            f"{len(camera_between)} < {n - 1}"
        )

    if len(camera_valid) < n - 1:
        raise ValueError(
            f"{seq}: Camera valid length too short: "
            f"{len(camera_valid)} < {n - 1}"
        )

    return SimpleNamespace(
        gps_local=gps_local,
        imu_acc=imu_acc,
        imu_gyro=imu_gyro,
        timestamps=timestamps,
        initial_oxts_rpy=initial_oxts_rpy,
        lidar_between=lidar_between,
        lidar_valid=lidar_valid,
        lidar_quality=lidar_quality,
        camera_between=camera_between,
        camera_valid=camera_valid,
        camera_quality=camera_quality,
    )



def initial_pose_from_oxts(gtsam, measurements):
    """
    Create the sequence initial body pose from frozen KITTI OXTS roll/pitch/yaw.
    Translation remains zero because Stage-7 local GPS is anchored at frame 0.
    Evaluation ground truth is not used here.
    """
    roll, pitch, yaw = map(float, measurements.initial_oxts_rpy)

    rotation = gtsam.Rot3.RzRyRx(
        roll,
        pitch,
        yaw,
    )

    return gtsam.Pose3(
        rotation,
        gtsam.Point3(0.0, 0.0, 0.0),
    )


def initial_trajectory_from_pose(
    gtsam,
    measurements,
    initial_pose,
):
    """
    Preserve the existing fair initialization policy for all cases:
    LiDAR relative odometry, then camera, then identity fallback.
    The chain now starts from measured OXTS attitude instead of identity.
    """
    n = len(measurements.gps_local)
    poses = [initial_pose]

    for i in range(n - 1):
        if bool(measurements.lidar_valid[i]):
            delta = _pose3_from_matrix(
                gtsam,
                measurements.lidar_between[i],
            )
        elif bool(measurements.camera_valid[i]):
            delta = _pose3_from_matrix(
                gtsam,
                measurements.camera_between[i],
            )
        else:
            delta = gtsam.Pose3()

        poses.append(
            poses[-1].compose(delta)
        )

    return poses

def pred(seq, source, sensor, n):
    base = sequence_out(seq) / "predictions" / source

    reliability_path = base / f"{sensor}_reliability.txt"
    uncertainty_path = base / f"{sensor}_uncertainty.txt"

    if not reliability_path.exists():
        raise FileNotFoundError(reliability_path)

    if not uncertainty_path.exists():
        raise FileNotFoundError(uncertainty_path)

    r = np.loadtxt(reliability_path).reshape(-1)
    u = np.loadtxt(uncertainty_path).reshape(-1)

    if len(r) != n or len(u) != n:
        raise ValueError(
            f"{seq}/{source}/{sensor} length mismatch: "
            f"r={len(r)}, u={len(u)}, expected={n}"
        )

    if not np.isfinite(r).all():
        raise ValueError(
            f"{seq}/{source}/{sensor}: reliability contains NaN/Inf"
        )

    if not np.isfinite(u).all():
        raise ValueError(
            f"{seq}/{source}/{sensor}: uncertainty contains NaN/Inf"
        )

    return np.clip(r, 0.0, 1.0), np.clip(u, 0.0, 1.0)


def confidence(u):
    return float(
        np.clip(
            1.0 - float(u),
            0.0,
            1.0,
        )
    )


def gps_sigma(cfg, r, u):
    predictive_sigma = _gps_sigma(
        cfg,
        "predictive",
        float(r),
    )

    c = confidence(u)

    return GPS_FIXED + c * (
        predictive_sigma - GPS_FIXED
    )


def imu_sigma(cfg, r, u):
    predictive_sigma = (
        cfg.imu_rot_sigma_min
        + (1.0 - float(r))
        * (
            cfg.imu_rot_sigma_max
            - cfg.imu_rot_sigma_min
        )
    )

    c = confidence(u)

    return IMU_R0 + c * (
        predictive_sigma - IMU_R0
    )


def cam_sigma(cfg, r, u):
    predictive_rot, predictive_trans = _camera_sigmas(
        cfg,
        "predictive",
        float(r),
    )

    c = confidence(u)

    rot_sigma = CAM_R0 + c * (
        predictive_rot - CAM_R0
    )

    trans_sigma = CAM_T0 + c * (
        predictive_trans - CAM_T0
    )

    return rot_sigma, trans_sigma


def evaluate(seq, traj):
    gt_path = sequence_out(seq) / "ground_truth.txt"

    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gt = np.loadtxt(
        gt_path,
        dtype=float,
    )[:, :3]

    est = np.asarray(
        traj,
        dtype=float,
    )[:, :3]

    n = min(
        len(gt),
        len(est),
    )

    if n == 0:
        raise ValueError(
            f"{seq}: empty trajectory during evaluation"
        )

    e = est[:n] - gt[:n]

    d3 = np.linalg.norm(
        e,
        axis=1,
    )

    d2 = np.linalg.norm(
        e[:, :2],
        axis=1,
    )

    return {
        "ATE3D": float(
            np.sqrt(
                np.mean(d3 * d3)
            )
        ),
        "ATE2D": float(
            np.sqrt(
                np.mean(d2 * d2)
            )
        ),
        "Mean3D": float(
            np.mean(d3)
        ),
        "Max3D": float(
            np.max(d3)
        ),
        "ZRMSE": float(
            np.sqrt(
                np.mean(
                    e[:, 2] ** 2
                )
            )
        ),
    }


def run_case(
    seq,
    name,
    gps_src=None,
    imu_src=None,
    cam_src=None,
):
    import gtsam

    print()
    print("=" * 80)
    print(f"{seq} | {name}")
    print("=" * 80)

    m = load_measurements(seq)

    n = len(m.gps_local)

    cfg = FourSensorFactorConfig()

    cache = {}

    sources = set(
        x
        for x in (
            gps_src,
            imu_src,
            cam_src,
        )
        if x is not None
    )

    for source in sources:
        for sensor in (
            "gps",
            "imu",
            "camera",
        ):
            cache[(source, sensor)] = pred(
                seq,
                source,
                sensor,
                n,
            )

    initial_pose = initial_pose_from_oxts(
        gtsam,
        m,
    )

    poses0 = initial_trajectory_from_pose(
        gtsam,
        m,
        initial_pose,
    )

    times = np.asarray(
        m.timestamps,
        dtype=float,
    )

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = (
        gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [cfg.prior_rotation_sigma] * 3
                + [cfg.prior_translation_sigma] * 3,
                dtype=float,
            )
        )
    )

    graph.add(
        gtsam.PriorFactorPose3(
            0,
            initial_pose,
            prior_noise,
        )
    )

    for i, pose in enumerate(poses0):
        initial.insert(
            i,
            pose,
        )

    for i in range(n):
        if gps_src is None:
            gs = GPS_FIXED
        else:
            r, u = cache[
                (gps_src, "gps")
            ]

            gs = gps_sigma(
                cfg,
                r[i],
                u[i],
            )

        gps_pose = gtsam.Pose3(
            gtsam.Rot3(),
            gtsam.Point3(
                *map(
                    float,
                    m.gps_local[i, :3],
                )
            ),
        )

        gps_noise = (
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [1e6] * 3
                    + [gs] * 3,
                    dtype=float,
                )
            )
        )

        graph.add(
            gtsam.PriorFactorPose3(
                i,
                gps_pose,
                gps_noise,
            )
        )

        if i == n - 1:
            continue

        dt = float(
            times[i + 1]
            - times[i]
        )

        if (
            not np.isfinite(dt)
            or dt <= 0.0
            or dt > 1.0
        ):
            dt = 0.1

        if imu_src is None:
            ir = IMU_R0
        else:
            r, u = cache[
                (imu_src, "imu")
            ]

            ir = imu_sigma(
                cfg,
                r[i + 1],
                u[i + 1],
            )

        imu_measurement = _imu_between_pose(
            gtsam,
            m.imu_gyro[i],
            dt,
        )

        imu_noise = (
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [ir] * 3
                    + [cfg.imu_translation_sigma] * 3,
                    dtype=float,
                )
            )
        )

        graph.add(
            gtsam.BetweenFactorPose3(
                i,
                i + 1,
                imu_measurement,
                imu_noise,
            )
        )

        if bool(m.lidar_valid[i]):
            lidar_measurement = (
                _pose3_from_matrix(
                    gtsam,
                    m.lidar_between[i],
                )
            )

            lidar_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.array(
                        [LIDAR_R0] * 3
                        + [LIDAR_T0] * 3,
                        dtype=float,
                    )
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    lidar_measurement,
                    lidar_noise,
                )
            )

        if bool(m.camera_valid[i]):
            if cam_src is None:
                cr = CAM_R0
                ct = CAM_T0
            else:
                r, u = cache[
                    (cam_src, "camera")
                ]

                cr, ct = cam_sigma(
                    cfg,
                    r[i + 1],
                    u[i + 1],
                )

            camera_measurement = (
                _pose3_from_matrix(
                    gtsam,
                    m.camera_between[i],
                )
            )

            camera_noise = (
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.array(
                        [cr] * 3
                        + [ct] * 3,
                        dtype=float,
                    )
                )
            )

            graph.add(
                gtsam.BetweenFactorPose3(
                    i,
                    i + 1,
                    camera_measurement,
                    camera_noise,
                )
            )

    print(
        "initial_oxts_rpy_deg="
        + np.array2string(
            np.degrees(m.initial_oxts_rpy),
            precision=3,
        )
    )

    print(
        f"frames={n}, "
        f"lidar_valid={int(np.sum(m.lidar_valid[:n-1]))}, "
        f"camera_valid={int(np.sum(m.camera_valid[:n-1]))}"
    )

    params = gtsam.LevenbergMarquardtParams()

    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-7)

    optimizer = (
        gtsam.LevenbergMarquardtOptimizer(
            graph,
            initial,
            params,
        )
    )

    result = optimizer.optimize()

    poses = np.stack(
        [
            _pose3_to_matrix(
                result.atPose3(i)
            )
            for i in range(n)
        ]
    )

    traj = poses[:, :3, 3]

    od = (
        sequence_out(seq)
        / "factor_graph"
        / name
    )

    od.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        od / "poses.npy",
        poses,
    )

    np.savetxt(
        od / "trajectory.txt",
        traj,
        fmt="%.9f",
    )

    metrics = evaluate(
        seq,
        traj,
    )

    (
        od / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        seq,
        name,
        metrics,
    )

    return metrics


CASES = [
    (
        "S0_fixed",
        None,
        None,
        None,
    ),
    (
        "S1_v5_gru",
        "v5",
        "v5",
        "v5",
    ),
    (
        "S2_v6_all_mamba",
        "v6",
        "v6",
        "v6",
    ),
    (
        "S3_d5_hybrid",
        "v6",
        "v5",
        "v6",
    ),
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        required=True,
        choices=TEST_SEQUENCES,
    )

    args = parser.parse_args()

    rows = []

    for (
        name,
        gps_source,
        imu_source,
        camera_source,
    ) in CASES:
        metrics = run_case(
            args.sequence,
            name,
            gps_source,
            imu_source,
            camera_source,
        )

        rows.append(
            {
                "case": name,
                **metrics,
            }
        )

    fixed_ate = rows[0]["ATE3D"]

    for row in rows:
        row["vs_fixed_pct"] = (
            (fixed_ate - row["ATE3D"])
            / fixed_ate
            * 100.0
        )

    output = (
        sequence_out(args.sequence)
        / "stage7_results.csv"
    )

    with output.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("STAGE-7 SEQUENCE COMPLETE")
    print("=" * 80)

    for row in rows:
        print(
            f"{row['case']:18s} "
            f"ATE3D={row['ATE3D']:.6f} "
            f"ATE2D={row['ATE2D']:.6f} "
            f"ZRMSE={row['ZRMSE']:.6f} "
            f"vs_fixed={row['vs_fixed_pct']:+.2f}%"
        )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
