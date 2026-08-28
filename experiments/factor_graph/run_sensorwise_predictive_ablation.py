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

from src.factor_graph.sensorwise_reliability_graph import (
    ReliabilitySource,
    run_sensorwise_reliability_graph,
)


DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

PREDICTIVE_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "results",
    "sensorwise_predictive_ablation",
)


CASES = [
    (
        "fixed",
        set(),
    ),
    (
        "gps_only",
        {
            "gps",
        },
    ),
    (
        "imu_only",
        {
            "imu",
        },
    ),
    (
        "lidar_only",
        {
            "lidar",
        },
    ),
    (
        "camera_only",
        {
            "camera",
        },
    ),
    (
        "gps_imu",
        {
            "gps",
            "imu",
        },
    ),
    (
        "gps_lidar",
        {
            "gps",
            "lidar",
        },
    ),
    (
        "gps_camera",
        {
            "gps",
            "camera",
        },
    ),
    (
        "gps_imu_lidar",
        {
            "gps",
            "imu",
            "lidar",
        },
    ),
    (
        "gps_imu_camera",
        {
            "gps",
            "imu",
            "camera",
        },
    ),
    (
        "all_predictive_v1",
        {
            "gps",
            "imu",
            "lidar",
            "camera",
        },
    ),
]


def make_sources(
    predictive_sensors,
):
    return {
        sensor:
            ReliabilitySource(
                "predictive"
                if sensor
                in predictive_sensors
                else
                "fixed"
            )
        for sensor in (
            "gps",
            "imu",
            "lidar",
            "camera",
        )
    }


def main():
    measurements = (
        load_degraded_four_sensor_measurements(
            DEGRADED_DIR
        )
    )

    print("=" * 112)
    print(
        "SENSOR-WISE PREDICTIVE RELIABILITY ABLATION"
    )
    print("=" * 112)

    print(
        "Frames:",
        len(
            measurements.gps_local
        ),
    )

    for index, (
        name,
        active,
    ) in enumerate(
        CASES,
        start=1,
    ):
        print()
        print(
            "#" * 112
        )

        print(
            f"[{index}/{len(CASES)}] "
            f"{name}"
        )

        print(
            "Predictive sensors:",
            sorted(
                active
            ),
        )

        print(
            "#" * 112
        )

        output_dir = os.path.join(
            OUTPUT_ROOT,
            name,
        )

        run_sensorwise_reliability_graph(
            measurements,
            output_dir,
            make_sources(
                active
            ),
            predictive_dir=
                PREDICTIVE_DIR,
            oracle_path=
                None,
        )

    print()
    print(
        "Predictive sensor-wise ablation finished."
    )


if __name__ == "__main__":
    main()
