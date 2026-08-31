from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.oracle_factor_reliability import (
    OracleReliabilityConfig,
    compute_oracle_factor_reliability,
)

SENSORS = ("gps", "imu", "lidar", "camera")
HORIZON = 3


def _finite_mask_from_between(between: np.ndarray) -> np.ndarray:
    between = np.asarray(between, dtype=np.float64)
    if between.ndim != 3 or between.shape[1:] != (4, 4):
        raise ValueError(f"between must be (N,4,4), got {between.shape}")
    return np.all(np.isfinite(between), axis=(1, 2))


def _load_physical_lidar(path: Path):
    data = np.load(path, allow_pickle=False)

    if "body_between" not in data.files:
        raise KeyError(
            f"{path} has no body_between. Available: {data.files}"
        )

    between = np.asarray(data["body_between"], dtype=np.float64)

    valid = _finite_mask_from_between(between)

    # Prefer actual ICP convergence when present.
    if "converged" in data.files:
        converged = np.asarray(data["converged"]).reshape(-1).astype(bool)
        if len(converged) == len(valid):
            valid &= converged

    return between, valid


def _future_target(current: np.ndarray, horizon: int) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    n = len(current)
    out = np.empty(n, dtype=np.float32)

    if horizon < n:
        out[: n - horizon] = current[horizon:]
        out[n - horizon :] = current[-1]
    else:
        out[:] = current[-1]

    return out


def _stats(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    f = x[np.isfinite(x)]
    if len(f) == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(np.min(f)),
        "max": float(np.max(f)),
        "mean": float(np.mean(f)),
        "median": float(np.median(f)),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--corrected-features",
        default=str(
            ROOT
            / "results/predictive_factor_reliability_v2_corrected"
            / "factor_reliability_training_data.npz"
        ),
    )

    p.add_argument(
        "--degraded-dir",
        default=str(ROOT / "results/degraded_four_sensor_measurements"),
    )

    p.add_argument(
        "--physical-lidar",
        default=str(
            ROOT
            / "results/lidar_physical_oracle_benchmark"
            / "physical_lidar_factor_data.npz"
        ),
    )

    p.add_argument(
        "--reference-poses",
        default=str(ROOT / "results/four_sensor_fixed_fg/poses.npy"),
    )

    p.add_argument(
        "--output-dir",
        default=str(ROOT / "results/predictive_factor_reliability_v3_same_source"),
    )

    a = p.parse_args()

    corrected_path = Path(a.corrected_features)
    degraded_dir = Path(a.degraded_dir)
    physical_lidar_path = Path(a.physical_lidar)
    reference_poses_path = Path(a.reference_poses)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        corrected_path,
        degraded_dir / "degraded_sensor_data.npz",
        degraded_dir / "degraded_camera_factor_data.npz",
        physical_lidar_path,
        reference_poses_path,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    corrected = np.load(corrected_path, allow_pickle=False)
    sensor = np.load(
        degraded_dir / "degraded_sensor_data.npz",
        allow_pickle=False,
    )
    camera = np.load(
        degraded_dir / "degraded_camera_factor_data.npz",
        allow_pickle=False,
    )
    reference_poses = np.load(reference_poses_path, allow_pickle=False)

    lidar_between, lidar_valid = _load_physical_lidar(
        physical_lidar_path
    )

    camera_between = np.asarray(
        camera["between_measurements"],
        dtype=np.float64,
    )
    camera_valid = np.asarray(
        camera["valid"],
    ).reshape(-1).astype(bool)

    clean_gps_local = np.asarray(
        sensor["clean_gps_local"],
        dtype=np.float64,
    )
    degraded_gps_local = np.asarray(
        sensor["degraded_gps_local"],
        dtype=np.float64,
    )
    degraded_imu_gyro = np.asarray(
        sensor["degraded_imu_gyro"],
        dtype=np.float64,
    )
    timestamps_seconds = np.asarray(
        sensor["timestamps_seconds"],
        dtype=np.float64,
    ).reshape(-1)

    feature_lengths = {
        s: len(corrected[f"{s}_features"])
        for s in SENSORS
    }

    n = min(
        *feature_lengths.values(),
        len(clean_gps_local),
        len(degraded_gps_local),
        len(degraded_imu_gyro),
        len(timestamps_seconds),
        len(reference_poses),
        len(lidar_between) + 1,
        len(camera_between) + 1,
    )

    print("=" * 110)
    print("BUILD SAME-SOURCE FACTOR-ERROR RELIABILITY LABELS")
    print("=" * 110)
    print("Frames:", n)
    print("Corrected feature file:", corrected_path)
    print("GPS/IMU measurement source:", degraded_dir / "degraded_sensor_data.npz")
    print("Camera factor source:", degraded_dir / "degraded_camera_factor_data.npz")
    print("LiDAR factor source:", physical_lidar_path)
    print("Reference poses:", reference_poses_path)

    result = compute_oracle_factor_reliability(
        clean_gps_local=clean_gps_local[:n],
        degraded_gps_local=degraded_gps_local[:n],
        degraded_imu_gyro=degraded_imu_gyro[:n],
        timestamps_seconds=timestamps_seconds[:n],
        lidar_between=lidar_between[: n - 1],
        lidar_valid=lidar_valid[: n - 1],
        camera_between=camera_between[: n - 1],
        camera_valid=camera_valid[: n - 1],
        reference_poses=reference_poses[:n],
        config=OracleReliabilityConfig(),
    )

    payload = {
        "horizon": np.asarray([HORIZON], dtype=np.int64),
    }

    for s in SENSORS:
        x = np.asarray(
            corrected[f"{s}_features"][:n],
            dtype=np.float32,
        )

        current = np.asarray(
            result[f"{s}_reliability"][:n],
            dtype=np.float32,
        )
        current = np.clip(current, 0.01, 1.0)

        future = _future_target(current, HORIZON)

        payload[f"{s}_features"] = x
        payload[f"{s}_current_factor_reliability"] = current
        payload[f"{s}_future_factor_reliability"] = future

    # Diagnostics are saved so every label can be traced back to an actual
    # same-realization factor error.
    for key, value in result.items():
        payload[f"diagnostic_{key}"] = np.asarray(value[:n])

    output_path = out_dir / "factor_reliability_training_data.npz"
    np.savez_compressed(output_path, **payload)

    metadata = {
        "version": "v3_same_source",
        "frames": int(n),
        "horizon": HORIZON,
        "feature_source": str(corrected_path),
        "gps_imu_measurement_source": str(
            degraded_dir / "degraded_sensor_data.npz"
        ),
        "camera_factor_source": str(
            degraded_dir / "degraded_camera_factor_data.npz"
        ),
        "lidar_factor_source": str(physical_lidar_path),
        "reference_poses": str(reference_poses_path),
        "label_mapping": "OracleReliabilityConfig / actual factor error",
        "principle": (
            "Each training target is recomputed from the same degraded "
            "measurement/factor stream represented by the corrected features."
        ),
    }

    label_stats = {}
    error_stats = {}

    for s in SENSORS:
        label_stats[s] = _stats(
            payload[f"{s}_current_factor_reliability"]
        )

    error_stats["gps_error_m"] = _stats(
        result["gps_error_m"]
    )
    error_stats["imu_rotation_error_deg"] = _stats(
        result["imu_rotation_error_deg"]
    )
    error_stats["lidar_translation_error_m"] = _stats(
        result["lidar_translation_error_m"]
    )
    error_stats["lidar_rotation_error_deg"] = _stats(
        result["lidar_rotation_error_deg"]
    )
    error_stats["camera_translation_error_m"] = _stats(
        result["camera_translation_error_m"]
    )
    error_stats["camera_rotation_error_deg"] = _stats(
        result["camera_rotation_error_deg"]
    )

    metadata["label_stats"] = label_stats
    metadata["error_stats"] = error_stats

    (out_dir / "same_source_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    for s in SENSORS:
        y = payload[f"{s}_current_factor_reliability"]
        print(
            f"{s:8s} reliability "
            f"min/max/mean/median = "
            f"{np.min(y):.6f} / "
            f"{np.max(y):.6f} / "
            f"{np.mean(y):.6f} / "
            f"{np.median(y):.6f}"
        )

    print()
    print("LiDAR valid pairs:", int(np.sum(lidar_valid[: n - 1])), "/", n - 1)
    print("Camera valid pairs:", int(np.sum(camera_valid[: n - 1])), "/", n - 1)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
