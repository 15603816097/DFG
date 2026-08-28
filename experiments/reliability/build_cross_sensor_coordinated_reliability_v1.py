from __future__ import annotations

import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from src.reliability.cross_sensor_coordinator import (
    CoordinatorConfig,
    SENSORS,
    coordinate_reliabilities,
    coordination_diagnostics,
)


SOURCE_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "cross_sensor_coordinated_reliability_v1",
)


CANDIDATE_FILES = {
    "gps": [
        "gps_predictive_prior_target_aligned.txt",
        "gps_factor_future_source_aligned.txt",
        "gps_factor_current_prediction.txt",
    ],
    "imu": [
        "imu_predictive_prior_target_aligned.txt",
        "imu_factor_future_source_aligned.txt",
        "imu_factor_current_prediction.txt",
    ],
    "lidar": [
        "lidar_predictive_prior_target_aligned.txt",
        "lidar_factor_future_source_aligned.txt",
        "lidar_factor_current_prediction.txt",
    ],
    "camera": [
        "camera_predictive_prior_target_aligned.txt",
        "camera_factor_future_source_aligned.txt",
        "camera_factor_current_prediction.txt",
    ],
}


# The downstream factor graph has historically used target-aligned
# reliability files.  We write both common names so this package is
# compatible with the current DFG runners without touching V1 files.
OUTPUT_FILES = {
    "gps": [
        "gps_predictive_prior_target_aligned.txt",
        "gps_factor_future_source_aligned.txt",
    ],
    "imu": [
        "imu_predictive_prior_target_aligned.txt",
        "imu_factor_future_source_aligned.txt",
    ],
    "lidar": [
        "lidar_predictive_prior_target_aligned.txt",
        "lidar_factor_future_source_aligned.txt",
    ],
    "camera": [
        "camera_predictive_prior_target_aligned.txt",
        "camera_factor_future_source_aligned.txt",
    ],
}


def _load_first_existing(sensor: str):
    checked = []
    for name in CANDIDATE_FILES[sensor]:
        path = os.path.join(SOURCE_DIR, name)
        checked.append(path)
        if os.path.isfile(path):
            x = np.loadtxt(path, dtype=np.float64).reshape(-1)
            return x, path
    raise FileNotFoundError(
        "Could not find target-aligned reliability for "
        f"{sensor}. Checked:\n  " + "\n  ".join(checked)
    )


def main():
    print("=" * 108)
    print("BUILD SENSOR-SPECIFIC + CROSS-SENSOR COORDINATED RELIABILITY V1")
    print("=" * 108)
    print("Source:", SOURCE_DIR)
    print("Output:", OUTPUT_DIR)
    print()
    print("IMPORTANT:")
    print("  - Existing V1 predictions are READ ONLY.")
    print("  - No old result directory is overwritten.")
    print("  - This stage coordinates reliability before covariance mapping.")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    original = {}
    source_files = {}

    for sensor in SENSORS:
        original[sensor], source_files[sensor] = _load_first_existing(sensor)

    n = len(original["gps"])
    for sensor in SENSORS:
        if len(original[sensor]) != n:
            raise RuntimeError(
                f"Length mismatch: gps={n}, "
                f"{sensor}={len(original[sensor])}"
            )

    cfg = CoordinatorConfig()
    coordinated = coordinate_reliabilities(
        original,
        config=cfg,
    )

    diagnostics = coordination_diagnostics(
        original,
        coordinated,
    )

    for sensor in SENSORS:
        for filename in OUTPUT_FILES[sensor]:
            np.savetxt(
                os.path.join(OUTPUT_DIR, filename),
                coordinated[sensor],
                fmt="%.10f",
            )

        # Keep the untouched original prediction for audit/debug only.
        np.savetxt(
            os.path.join(
                OUTPUT_DIR,
                f"{sensor}_original_v1_target_aligned.txt",
            ),
            original[sensor],
            fmt="%.10f",
        )

    # A single NPZ makes later plotting/ablation easy.
    np.savez_compressed(
        os.path.join(
            OUTPUT_DIR,
            "coordinated_reliability_v1.npz",
        ),
        **{
            **{
                f"{s}_original": original[s]
                for s in SENSORS
            },
            **{
                f"{s}_coordinated": coordinated[s]
                for s in SENSORS
            },
        },
    )

    metadata = {
        "frames": n,
        "source_dir": SOURCE_DIR,
        "source_files": source_files,
        "config": cfg.__dict__,
        "diagnostics": diagnostics,
    }
    with open(
        os.path.join(OUTPUT_DIR, "coordination_metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print(f"Frames: {n}")
    print("-" * 108)
    for sensor in SENSORS:
        d = diagnostics[sensor]
        print(
            f"{sensor.upper():7s} "
            f"V1 mean={d['original_mean']:.4f} "
            f"-> coordinated mean={d['coordinated_mean']:.4f} | "
            f"mean|delta|={d['mean_abs_change']:.4f} "
            f"max|delta|={d['max_abs_change']:.4f}"
        )

    print()
    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
