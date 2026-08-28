from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(
        0,
        ROOT,
    )


# IMPORTANT:
# The degraded measurement chain has different filenames from the clean
# body_relative_odometry directory:
#
#   degraded_lidar_factor_data.npz
#   degraded_camera_factor_data.npz
#
# Therefore we MUST use the dedicated degraded-measurement loader.
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)

from src.factor_graph.four_sensor_graph import (
    run_four_sensor_factor_graph,
)


DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

PREDICTION_DIR = os.path.join(
    ROOT,
    "results",
    "cross_sensor_coordinated_reliability_v1",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "cross_sensor_coordinated_fg_v1",
)


REQUIRED_PREDICTION_FILES = (
    "gps_predictive_prior_target_aligned.txt",
    "imu_predictive_prior_target_aligned.txt",
    "lidar_predictive_prior_target_aligned.txt",
    "camera_predictive_prior_target_aligned.txt",
)


def _check_inputs():
    if not os.path.isdir(
        DEGRADED_DIR
    ):
        raise FileNotFoundError(
            "Missing degraded measurement directory:\n"
            f"{DEGRADED_DIR}\n\n"
            "Run the degraded measurement builder first."
        )

    if not os.path.isdir(
        PREDICTION_DIR
    ):
        raise FileNotFoundError(
            "Missing coordinated reliability directory:\n"
            f"{PREDICTION_DIR}\n\n"
            "Run:\n"
            "python experiments/reliability/"
            "build_cross_sensor_coordinated_reliability_v1.py"
        )

    missing = []

    for filename in REQUIRED_PREDICTION_FILES:
        path = os.path.join(
            PREDICTION_DIR,
            filename,
        )

        if not os.path.isfile(
            path
        ):
            missing.append(
                path
            )

    if missing:
        raise FileNotFoundError(
            "Missing coordinated reliability file(s):\n"
            +
            "\n".join(
                missing
            )
        )


def main():
    print(
        "=" * 108
    )

    print(
        "SENSOR-SPECIFIC + CROSS-SENSOR "
        "COORDINATED FOUR-SENSOR FG V1 - FIX"
    )

    print(
        "=" * 108
    )

    _check_inputs()

    # ------------------------------------------------------------
    # FIX:
    # Do NOT call load_four_sensor_measurements(SEQUENCE, DEGRADED_DIR).
    #
    # That clean-data loader expects:
    #     lidar_factor_data.npz
    #     camera_factor_data.npz
    #
    # but the physically degraded chain stores:
    #     degraded_lidar_factor_data.npz
    #     degraded_camera_factor_data.npz
    #
    # The dedicated loader below is the same loader already used by
    # the successful degraded fixed/predictive/oracle experiments.
    # ------------------------------------------------------------
    measurements = (
        load_degraded_four_sensor_measurements(
            DEGRADED_DIR
        )
    )

    n_frames = len(
        measurements.gps_local
    )

    print(
        "Frames:",
        n_frames,
    )

    print(
        "Degraded measurement dir:",
        DEGRADED_DIR,
    )

    print(
        "Prediction dir:",
        PREDICTION_DIR,
    )

    print(
        "Output dir:",
        OUTPUT_DIR,
    )

    # Basic consistency checks before building a large graph.
    if len(
        measurements.imu_gyro
    ) != n_frames:
        raise RuntimeError(
            "GPS/IMU frame-count mismatch: "
            f"GPS={n_frames}, "
            f"IMU={len(measurements.imu_gyro)}"
        )

    if len(
        measurements.lidar_between
    ) != n_frames - 1:
        raise RuntimeError(
            "LiDAR relative-factor count mismatch: "
            f"expected={n_frames - 1}, "
            f"actual={len(measurements.lidar_between)}"
        )

    if len(
        measurements.camera_between
    ) != n_frames - 1:
        raise RuntimeError(
            "Camera relative-factor count mismatch: "
            f"expected={n_frames - 1}, "
            f"actual={len(measurements.camera_between)}"
        )

    trajectory, _, counts = (
        run_four_sensor_factor_graph(
            measurements,
            OUTPUT_DIR,
            mode="predictive",
            prediction_dir=
                PREDICTION_DIR,
            use_gps=True,
            use_imu=True,
            use_lidar=True,
            use_camera=True,
        )
    )

    print(
        "Trajectory:",
        trajectory.shape,
    )

    print(
        "Factor counts:",
        counts,
    )

    print(
        "Final position:",
        trajectory[
            -1
        ],
    )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
