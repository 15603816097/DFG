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


from src.factor_graph.four_sensor_graph import (
    load_four_sensor_measurements,
    run_four_sensor_factor_graph,
)


SEQUENCE = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)

BODY_DIR = os.path.join(
    ROOT,
    "results",
    "body_relative_odometry",
)

PREDICTION_DIR = os.path.join(
    ROOT,
    "results",
    "multisensor_predictive_reliability_v2",
)

OUTPUT = os.path.join(
    ROOT,
    "results",
    "four_sensor_predictive_fg",
)


def main():
    print("=" * 96)
    print(
        "FOUR-SENSOR PREDICTIVE RELIABILITY FACTOR GRAPH"
    )
    print("=" * 96)

    measurements = load_four_sensor_measurements(
        SEQUENCE,
        BODY_DIR,
    )

    print(
        "Frames:",
        len(
            measurements.gps_local
        ),
    )

    trajectory, _, _ = (
        run_four_sensor_factor_graph(
            measurements,
            OUTPUT,
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
        "Final position:",
        trajectory[
            -1
        ],
    )

    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
