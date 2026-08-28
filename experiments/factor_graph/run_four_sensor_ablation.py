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


CASES = [
    (
        "gps_imu",
        True,
        True,
        False,
        False,
    ),
    (
        "gps_imu_lidar",
        True,
        True,
        True,
        False,
    ),
    (
        "gps_imu_camera",
        True,
        True,
        False,
        True,
    ),
    (
        "gps_imu_lidar_camera",
        True,
        True,
        True,
        True,
    ),
]


def main():
    measurements = load_four_sensor_measurements(
        SEQUENCE,
        BODY_DIR,
    )

    print("=" * 100)
    print(
        "FOUR-SENSOR FIXED-COVARIANCE SENSOR CONTRIBUTION ABLATION"
    )
    print("=" * 100)

    for index, (
        name,
        use_gps,
        use_imu,
        use_lidar,
        use_camera,
    ) in enumerate(
        CASES,
        start=1,
    ):
        print()
        print(
            "#" * 100
        )
        print(
            f"[{index}/{len(CASES)}] {name}"
        )
        print(
            "#" * 100
        )

        output = os.path.join(
            ROOT,
            "results",
            "four_sensor_ablation",
            name,
        )

        run_four_sensor_factor_graph(
            measurements,
            output,
            mode="fixed",
            use_gps=
                use_gps,
            use_imu=
                use_imu,
            use_lidar=
                use_lidar,
            use_camera=
                use_camera,
        )

    print()
    print(
        "Ablation finished."
    )


if __name__ == "__main__":
    main()
