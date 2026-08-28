from __future__ import annotations
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.features.factor_oriented_features import (
    build_gps_factor_features,
    build_imu_factor_features,
    build_lidar_factor_features,
    build_camera_factor_features,
)

BASE_FEATURE_PATH = os.path.join(ROOT, "results", "multisensor_reliability_v2", "multisensor_reliability_data_v2.npz")
DEGRADED_SENSOR_PATH = os.path.join(ROOT, "results", "degraded_four_sensor_measurements", "degraded_sensor_data.npz")
DEGRADED_LIDAR_PATH = os.path.join(ROOT, "results", "degraded_four_sensor_measurements", "degraded_lidar_factor_data.npz")
DEGRADED_CAMERA_PATH = os.path.join(ROOT, "results", "degraded_four_sensor_measurements", "degraded_camera_factor_data.npz")
ORACLE_PATH = os.path.join(ROOT, "results", "oracle_factor_reliability", "oracle_factor_reliability.npz")
OUTPUT_DIR = os.path.join(ROOT, "results", "predictive_factor_reliability_v2")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "factor_reliability_v2_data.npz")
HORIZON = 3

def make_future(current, horizon):
    current = np.asarray(current, dtype=np.float32)
    n = len(current)
    future = np.empty(n, dtype=np.float32)
    future[:-horizon] = current[horizon:]
    future[-horizon:] = current[-1]
    return future

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base = np.load(BASE_FEATURE_PATH, allow_pickle=False)
    degraded_sensor = np.load(DEGRADED_SENSOR_PATH, allow_pickle=False)
    degraded_lidar = np.load(DEGRADED_LIDAR_PATH, allow_pickle=False)
    degraded_camera = np.load(DEGRADED_CAMERA_PATH, allow_pickle=False)
    oracle = np.load(ORACLE_PATH, allow_pickle=False)

    features = {
        "gps": build_gps_factor_features(
            base["gps_features"],
            degraded_sensor["degraded_gps_local"],
            degraded_sensor["timestamps_seconds"],
            window=10,
        ),
        "imu": build_imu_factor_features(
            base["imu_features"],
            degraded_sensor["degraded_imu_acc"],
            degraded_sensor["degraded_imu_gyro"],
            degraded_sensor["timestamps_seconds"],
            window=10,
        ),
        "lidar": build_lidar_factor_features(
            base["lidar_features"],
            degraded_lidar,
            window=10,
        ),
        "camera": build_camera_factor_features(
            base["camera_features"],
            degraded_camera,
            window=10,
        ),
    }

    n = min(
        *(len(v) for v in features.values()),
        *(len(oracle[f"{s}_reliability"]) for s in ("gps", "imu", "lidar", "camera")),
    )

    payload = {
        "horizon": np.asarray([HORIZON], dtype=np.int64),
    }

    for sensor in ("gps", "imu", "lidar", "camera"):
        current = np.asarray(oracle[f"{sensor}_reliability"][:n], dtype=np.float32)
        payload[f"{sensor}_features"] = features[sensor][:n]
        payload[f"{sensor}_current_label"] = current
        payload[f"{sensor}_future_label"] = make_future(current, HORIZON)

    np.savez_compressed(OUTPUT_PATH, **payload)

    print("=" * 108)
    print("FACTOR-ORIENTED PREDICTIVE RELIABILITY V2 DATA")
    print("=" * 108)
    print("Frames:", n)
    print("Horizon:", HORIZON)

    for sensor in ("gps", "imu", "lidar", "camera"):
        y = payload[f"{sensor}_current_label"]
        print(
            f"{sensor:8s} features={payload[f'{sensor}_features'].shape} "
            f"reliability min/max/mean={y.min():.4f}/{y.max():.4f}/{y.mean():.4f}"
        )

    print("Saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
