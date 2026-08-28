"""
Predictive Reliability Dataset V3
=================================

Core idea
---------
For target frame t:

    current label:
        R_current(t)

    future label:
        R_future(t + H)

The model input is a causal sequence ending at t.

During online deployment, the future head prediction generated at frame
(t-H) is stored and aligned to frame t.  Therefore the factor graph at
frame t uses a reliability prediction made H frames earlier.

This file uses the existing 30-D causal health features and never uses
ground truth as a model input.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.loader.imu_loader import IMULoader
from src.reliability.health_features import build_health_features


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "kitti"
    / "2011_10_03"
    / "2011_10_03_drive_0027_sync"
)

DEFAULT_GPS_PATH = (
    PROJECT_ROOT
    / "results"
    / "progressive_gps_degradation"
    / "gps_corrupted.txt"
)

DEFAULT_CURRENT_LABEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictive_reliability_labels"
    / "current_reliability.txt"
)

DEFAULT_FUTURE_LABEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictive_reliability_labels"
    / "future_window_reliability_h5.txt"
)


def load_imu(
    sequence_path=DEFAULT_DATASET,
):
    loader = IMULoader(
        sequence_path
    )

    acceleration = []
    gyro = []

    for i in range(
        len(loader)
    ):
        item = loader[i]

        acceleration.append(
            item["acceleration"]
        )

        gyro.append(
            item["angular_velocity"]
        )

    return (
        np.asarray(
            acceleration,
            dtype=np.float64,
        ),
        np.asarray(
            gyro,
            dtype=np.float64,
        ),
    )


def build_aligned_data(
    dataset_path=DEFAULT_DATASET,
    gps_path=DEFAULT_GPS_PATH,
    current_label_path=
        DEFAULT_CURRENT_LABEL_PATH,
    future_label_path=
        DEFAULT_FUTURE_LABEL_PATH,
    health_window=10,
):
    gps = np.loadtxt(
        gps_path,
        dtype=np.float64,
    )

    current_label = np.loadtxt(
        current_label_path,
        dtype=np.float64,
    ).reshape(-1)

    future_label = np.loadtxt(
        future_label_path,
        dtype=np.float64,
    ).reshape(-1)

    acceleration, gyro = load_imu(
        dataset_path
    )

    n = min(
        len(gps),
        len(current_label),
        len(future_label),
        len(acceleration),
        len(gyro),
    )

    gps = gps[:n]
    acceleration = acceleration[:n]
    gyro = gyro[:n]
    current_label = current_label[:n]
    future_label = future_label[:n]

    features = build_health_features(
        gps,
        acceleration,
        gyro,
        dt=0.1,
        window=health_window,
    )

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    current_label = np.clip(
        current_label,
        0.0,
        1.0,
    ).astype(
        np.float32
    )

    future_label = np.clip(
        future_label,
        0.0,
        1.0,
    ).astype(
        np.float32
    )

    return (
        features,
        current_label,
        future_label,
    )


class DualHeadReliabilityDataset(
    Dataset
):
    """
    Each item:
        sequence:
            T x D

        current_label:
            R_current(t)

        future_label:
            R_future(t+H-style target generated in label file)

        frame_id:
            t
    """

    def __init__(
        self,
        features,
        current_labels,
        future_labels,
        frame_ids,
        sequence_length=64,
    ):
        self.features = torch.from_numpy(
            np.asarray(
                features,
                dtype=np.float32,
            )
        )

        self.current_labels = torch.from_numpy(
            np.asarray(
                current_labels,
                dtype=np.float32,
            )
        )

        self.future_labels = torch.from_numpy(
            np.asarray(
                future_labels,
                dtype=np.float32,
            )
        )

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
        frame = int(
            self.frame_ids[index]
        )

        start = (
            frame
            -
            self.sequence_length
            +
            1
        )

        if start < 0:
            raise IndexError(
                "frame_ids must start at "
                "sequence_length-1"
            )

        sequence = self.features[
            start:
            frame + 1
        ]

        return {
            "feature":
                sequence,

            "current_label":
                self.current_labels[
                    frame
                ],

            "future_label":
                self.future_labels[
                    frame
                ],

            "frame_id":
                frame,
        }
