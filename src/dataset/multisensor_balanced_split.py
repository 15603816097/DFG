"""
Balanced chronological split utilities.

The sequence remains chronological:
    Train = first 70%
    Val   = next 15%
    Test  = final 15%

The difference from V1 is that degradation generation itself is balanced
inside every split.
"""

from __future__ import annotations

import numpy as np


SENSORS = (
    "gps",
    "imu",
    "lidar",
    "camera",
)


def split_bounds(
    n_frames,
):
    n = int(
        n_frames
    )

    train_end = int(
        n
        *
        0.70
    )

    val_end = int(
        n
        *
        0.85
    )

    return {
        "train":
            (
                0,
                train_end,
            ),

        "val":
            (
                train_end,
                val_end,
            ),

        "test":
            (
                val_end,
                n,
            ),
    }


def distribution_summary(
    current_labels,
    future_labels,
):
    n = min(
        len(
            current_labels[
                "gps"
            ]
        ),
        len(
            future_labels[
                "gps"
            ]
        ),
    )

    bounds = split_bounds(
        n
    )

    result = {}

    for split_name, (
        start,
        end,
    ) in bounds.items():

        result[
            split_name
        ] = {}

        for sensor in SENSORS:
            current = np.asarray(
                current_labels[
                    sensor
                ][
                    start:end
                ],
                dtype=np.float64,
            )

            future = np.asarray(
                future_labels[
                    sensor
                ][
                    start:end
                ],
                dtype=np.float64,
            )

            result[
                split_name
            ][
                sensor
            ] = {
                "count":
                    int(
                        len(
                            future
                        )
                    ),

                "current_mean":
                    float(
                        current.mean()
                    ),

                "future_mean":
                    float(
                        future.mean()
                    ),

                "bad_ratio_current":
                    float(
                        np.mean(
                            current
                            <
                            0.5
                        )
                    ),

                "bad_ratio_future":
                    float(
                        np.mean(
                            future
                            <
                            0.5
                        )
                    ),

                "severe_ratio_future":
                    float(
                        np.mean(
                            future
                            <
                            0.2
                        )
                    ),
            }

    return result
