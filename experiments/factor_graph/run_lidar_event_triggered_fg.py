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


from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)

from src.factor_graph.lidar_event_triggered_covariance import (
    run_lidar_event_triggered_factor_graph,
)


DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

PREDICTION_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

EVENT_DATA_PATH = os.path.join(
    ROOT,
    "results",
    "lidar_event_triggered_reliability",
    "lidar_event_trigger.npz",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "lidar_event_triggered_fg",
)


def main():
    print(
        "=" * 112
    )

    print(
        "LIDAR EVENT-TRIGGERED DYNAMIC COVARIANCE FACTOR GRAPH"
    )

    print(
        "=" * 112
    )

    if not os.path.isfile(
        EVENT_DATA_PATH
    ):
        raise FileNotFoundError(
            "Run build_lidar_event_triggered_reliability.py first:\n"
            f"{EVENT_DATA_PATH}"
        )

    measurements = (
        load_degraded_four_sensor_measurements(
            DEGRADED_DIR
        )
    )

    print(
        "Frames:",
        len(
            measurements.gps_local
        ),
    )

    print(
        "Architecture:"
    )

    print(
        "  GPS     predictive"
    )

    print(
        "  IMU     predictive"
    )

    print(
        "  Camera  predictive"
    )

    print(
        "  LiDAR   fixed unless event triggered"
    )

    (
        trajectory,
        _,
        counts,
        triggered_count,
    ) = run_lidar_event_triggered_factor_graph(
        measurements=
            measurements,
        output_dir=
            OUTPUT_DIR,
        prediction_dir=
            PREDICTION_DIR,
        event_data_path=
            EVENT_DATA_PATH,
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
        "Triggered LiDAR factors:",
        triggered_count,
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
