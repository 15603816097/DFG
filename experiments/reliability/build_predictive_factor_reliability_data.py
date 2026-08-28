from __future__ import annotations

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


from src.dataset.factor_reliability_dataset import (
    SENSORS,
    load_factor_reliability_training_data,
)


FEATURE_PATH = os.path.join(
    ROOT,
    "results",
    "multisensor_reliability_v2",
    "multisensor_reliability_data_v2.npz",
)

ORACLE_PATH = os.path.join(
    ROOT,
    "results",
    "oracle_factor_reliability",
    "oracle_factor_reliability.npz",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "factor_reliability_training_data.npz",
)

HORIZON = 3


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    data = load_factor_reliability_training_data(
        FEATURE_PATH,
        ORACLE_PATH,
        horizon=HORIZON,
    )

    payload = {
        "horizon":
            np.asarray(
                [
                    data.horizon
                ],
                dtype=np.int64,
            ),
    }

    for sensor in SENSORS:
        payload[
            f"{sensor}_features"
        ] = data.features[
            sensor
        ]

        payload[
            f"{sensor}_current_factor_reliability"
        ] = data.current_labels[
            sensor
        ]

        payload[
            f"{sensor}_future_factor_reliability"
        ] = data.future_labels[
            sensor
        ]

    np.savez_compressed(
        OUTPUT_PATH,
        **payload,
    )

    print("=" * 100)
    print(
        "PREDICTIVE FACTOR RELIABILITY TRAINING DATA V1"
    )
    print("=" * 100)

    print(
        "Frames:",
        data.length,
    )

    print(
        "Horizon:",
        data.horizon,
    )

    for sensor in SENSORS:
        current = data.current_labels[
            sensor
        ]

        future = data.future_labels[
            sensor
        ]

        print()
        print(
            sensor.upper()
        )

        print(
            "features:",
            data.features[
                sensor
            ].shape,
        )

        print(
            "current reliability "
            f"min/max/mean: "
            f"{current.min():.4f} / "
            f"{current.max():.4f} / "
            f"{current.mean():.4f}"
        )

        print(
            "future reliability  "
            f"min/max/mean: "
            f"{future.min():.4f} / "
            f"{future.max():.4f} / "
            f"{future.mean():.4f}"
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
