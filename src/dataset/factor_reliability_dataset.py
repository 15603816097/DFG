from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SENSORS = (
    "gps",
    "imu",
    "lidar",
    "camera",
)


@dataclass
class FactorReliabilityData:
    features: dict
    current_labels: dict
    future_labels: dict
    horizon: int
    length: int


def load_factor_reliability_training_data(
    feature_npz_path,
    oracle_npz_path,
    horizon=3,
):
    """
    Build training targets for:

        X_t  -> current factor reliability at t
        X_t  -> future factor reliability at t + H

    The feature file is the existing Balanced V2 multi-sensor feature file.
    The oracle file contains factor-error-derived reliabilities.

    No old file is modified.
    """
    feature_npz_path = Path(
        feature_npz_path
    )

    oracle_npz_path = Path(
        oracle_npz_path
    )

    feature_data = np.load(
        feature_npz_path,
        allow_pickle=False,
    )

    oracle_data = np.load(
        oracle_npz_path,
        allow_pickle=False,
    )

    features = {}

    current_labels = {}

    future_labels = {}

    lengths = []

    for sensor in SENSORS:
        x = np.asarray(
            feature_data[
                f"{sensor}_features"
            ],
            dtype=np.float32,
        )

        y = np.asarray(
            oracle_data[
                f"{sensor}_reliability"
            ],
            dtype=np.float32,
        ).reshape(-1)

        features[
            sensor
        ] = x

        current_labels[
            sensor
        ] = np.clip(
            y,
            0.0,
            1.0,
        )

        lengths.extend(
            [
                len(
                    x
                ),
                len(
                    y
                ),
            ]
        )

    n = min(
        lengths
    )

    horizon = int(
        horizon
    )

    if horizon < 1:
        raise ValueError(
            "horizon must be >= 1"
        )

    for sensor in SENSORS:
        features[
            sensor
        ] = features[
            sensor
        ][
            :n
        ]

        current_labels[
            sensor
        ] = current_labels[
            sensor
        ][
            :n
        ]

        future = np.empty(
            n,
            dtype=np.float32,
        )

        if horizon < n:
            future[
                :n - horizon
            ] = current_labels[
                sensor
            ][
                horizon:
            ]

            future[
                n - horizon:
            ] = current_labels[
                sensor
            ][
                -1
            ]

        else:
            future[:] = current_labels[
                sensor
            ][
                -1
            ]

        future_labels[
            sensor
        ] = future

    return FactorReliabilityData(
        features=features,
        current_labels=current_labels,
        future_labels=future_labels,
        horizon=horizon,
        length=n,
    )


def compute_normalization(
    features,
    train_end,
):
    result = {}

    for sensor in SENSORS:
        x = np.asarray(
            features[
                sensor
            ][
                :train_end
            ],
            dtype=np.float32,
        )

        mean = x.mean(
            axis=0
        )

        std = x.std(
            axis=0
        )

        std[
            std
            <
            1e-6
        ] = 1.0

        result[
            sensor
        ] = {
            "mean":
                mean.astype(
                    np.float32
                ),
            "std":
                std.astype(
                    np.float32
                ),
        }

    return result


def apply_normalization(
    features,
    normalization,
):
    result = {}

    for sensor in SENSORS:
        result[
            sensor
        ] = (
            (
                np.asarray(
                    features[
                        sensor
                    ],
                    dtype=np.float32,
                )
                -
                normalization[
                    sensor
                ][
                    "mean"
                ]
            )
            /
            normalization[
                sensor
            ][
                "std"
            ]
        ).astype(
            np.float32
        )

    return result


class PredictiveFactorReliabilityDataset(
    Dataset
):
    def __init__(
        self,
        features,
        current_labels,
        future_labels,
        frame_ids,
        sequence_length=64,
    ):
        self.features = {
            sensor:
                torch.from_numpy(
                    np.asarray(
                        features[
                            sensor
                        ],
                        dtype=np.float32,
                    )
                )
            for sensor in SENSORS
        }

        self.current_labels = {
            sensor:
                torch.from_numpy(
                    np.asarray(
                        current_labels[
                            sensor
                        ],
                        dtype=np.float32,
                    )
                )
            for sensor in SENSORS
        }

        self.future_labels = {
            sensor:
                torch.from_numpy(
                    np.asarray(
                        future_labels[
                            sensor
                        ],
                        dtype=np.float32,
                    )
                )
            for sensor in SENSORS
        }

        self.frame_ids = np.asarray(
            frame_ids,
            dtype=np.int64,
        )

        self.sequence_length = int(
            sequence_length
        )

    def __len__(
        self
    ):
        return len(
            self.frame_ids
        )

    def __getitem__(
        self,
        index,
    ):
        frame_id = int(
            self.frame_ids[
                index
            ]
        )

        start = (
            frame_id
            -
            self.sequence_length
            +
            1
        )

        if start < 0:
            raise IndexError(
                "frame_ids must start at sequence_length - 1"
            )

        item = {
            "frame_id":
                frame_id,
        }

        for sensor in SENSORS:
            item[
                f"{sensor}_feature"
            ] = self.features[
                sensor
            ][
                start:
                frame_id + 1
            ]

            item[
                f"{sensor}_current_label"
            ] = self.current_labels[
                sensor
            ][
                frame_id
            ]

            item[
                f"{sensor}_future_label"
            ] = self.future_labels[
                sensor
            ][
                frame_id
            ]

        return item
