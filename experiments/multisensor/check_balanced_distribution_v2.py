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
    sys.path.insert(
        0,
        ROOT,
    )


from src.dataset.multisensor_balanced_split import (
    distribution_summary,
)


DATA_PATH = os.path.join(
    ROOT,
    "results",
    "multisensor_reliability_v2",
    "multisensor_reliability_data_v2.npz",
)


SENSORS = (
    "gps",
    "imu",
    "lidar",
    "camera",
)


def main():
    data = np.load(
        DATA_PATH
    )

    current = {
        sensor:
            data[
                f"{sensor}_current_label"
            ]
        for sensor in SENSORS
    }

    future = {
        sensor:
            data[
                f"{sensor}_future_label"
            ]
        for sensor in SENSORS
    }

    summary = (
        distribution_summary(
            current,
            future,
        )
    )

    print("=" * 112)
    print(
        "BALANCED TRAIN / VAL / TEST "
        "DISTRIBUTION CHECK V2"
    )
    print("=" * 112)

    print(
        f'{"Split":>8s}'
        f'{"Sensor":>10s}'
        f'{"FutureMean":>14s}'
        f'{"Bad<0.5":>12s}'
        f'{"Severe<0.2":>14s}'
    )

    print(
        "-" * 112
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        for sensor in SENSORS:
            row = summary[
                split
            ][
                sensor
            ]

            print(
                f"{split:>8s}"
                f"{sensor:>10s}"
                f"{row['future_mean']:14.4f}"
                f"{row['bad_ratio_future']:12.4f}"
                f"{row['severe_ratio_future']:14.4f}"
            )

    print()
    print(
        "Interpretation:"
    )

    print(
        "Each split should contain non-zero "
        "bad and severe ratios for every sensor."
    )


if __name__ == "__main__":
    main()
