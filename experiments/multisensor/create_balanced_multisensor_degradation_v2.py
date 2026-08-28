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


from src.loader.imu_loader import IMULoader

from src.multisensor_reliability.degradation.balanced_plans import (
    create_balanced_multisensor_degradation_plan,
    save_balanced_plan,
)


DATASET = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "multisensor_reliability_v2",
)


def main():
    n = len(
        IMULoader(
            DATASET
        )
    )

    plan = (
        create_balanced_multisensor_degradation_plan(
            n_frames=n,
            seed=20260826,
        )
    )

    save_balanced_plan(
        plan,
        OUTPUT_DIR,
    )

    print("=" * 92)
    print(
        "BALANCED MULTI-SENSOR "
        "DEGRADATION PLAN V2"
    )
    print("=" * 92)

    print(
        "Frames:",
        n,
    )

    for split_name, bounds in plan[
        "split_bounds"
    ].items():
        print(
            split_name,
            ":",
            bounds,
        )

    for sensor in (
        "gps",
        "imu",
        "lidar",
        "camera",
    ):
        x = plan[
            "severity"
        ][
            sensor
        ]

        print(
            f"{sensor:8s} "
            f"min={x.min():.3f} "
            f"max={x.max():.3f} "
            f"mean={x.mean():.3f}"
        )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
