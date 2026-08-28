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
    "predictive_factor_reliability_v1",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "predicted_factor_reliability_fg",
)


def main():
    print("=" * 108)
    print(
        "PREDICTED FUTURE FACTOR RELIABILITY FOUR-SENSOR FG"
    )
    print("=" * 108)

    measurements = load_degraded_four_sensor_measurements(
        DEGRADED_DIR
    )

    print(
        "Frames:",
        len(
            measurements.gps_local
        ),
    )

    trajectory, _, counts = run_four_sensor_factor_graph(
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
