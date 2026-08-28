"""
Frozen H=3 Mamba Inference on Multi-Degradation Scenarios
==========================================================

IMPORTANT:
    The model is NOT retrained for any new scenario.

Frozen model:
    results/horizon_sensitivity/h3/best_model.pt

Frozen normalization:
    results/horizon_sensitivity/h3/normalization.npz

For each scenario:
    build causal health features
    ->
    frozen dual-head Mamba
    ->
    source-frame future prediction
    ->
    target alignment:
        source t -> target t+3

Output:
    results/multi_degradation/<scenario>/
        predictive_prior.txt
        current_prediction.txt
        future_prediction_source_aligned.txt
        prediction_source_frame.txt
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
from src.reliability.health_features import (
    build_health_features,
)
from src.model.predictive_reliability_model import (
    PredictiveReliabilityModel,
)


DATASET = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)

MODEL_DIR = os.path.join(
    ROOT,
    "results",
    "horizon_sensitivity",
    "h3",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pt",
)

NORMALIZATION_PATH = os.path.join(
    MODEL_DIR,
    "normalization.npz",
)

SCENARIO_ROOT = os.path.join(
    ROOT,
    "results",
    "multi_degradation",
)

SCENARIOS = [
    "mild_progressive",
    "medium_progressive",
    "severe_progressive",
    "sudden",
    "bias_drift",
    "intermittent_outlier",
]

HORIZON = 3
SEQUENCE_LENGTH = 64
BATCH_SIZE = 128


class InferenceDataset(
    Dataset
):
    def __init__(
        self,
        features,
        frame_ids,
    ):
        self.features = torch.from_numpy(
            np.asarray(
                features,
                dtype=np.float32,
            )
        )

        self.frame_ids = np.asarray(
            frame_ids,
            dtype=np.int64,
        )

    def __len__(self):
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
            SEQUENCE_LENGTH
            +
            1
        )

        return {
            "feature":
                self.features[
                    start:
                    frame + 1
                ],

            "frame_id":
                frame,
        }


def load_imu():
    loader = IMULoader(
        DATASET
    )

    acceleration = []
    gyro = []

    for i in range(
        len(loader)
    ):
        item = loader[i]

        acceleration.append(
            item[
                "acceleration"
            ]
        )

        gyro.append(
            item[
                "angular_velocity"
            ]
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


@torch.no_grad()
def infer_scenario(
    model,
    device,
    mean,
    std,
    acceleration,
    gyro,
    scenario,
):
    folder = os.path.join(
        SCENARIO_ROOT,
        scenario,
    )

    gps_path = os.path.join(
        folder,
        "gps_corrupted.txt",
    )

    if not os.path.exists(
        gps_path
    ):
        raise FileNotFoundError(
            gps_path
        )

    gps = np.loadtxt(
        gps_path,
        dtype=np.float64,
    )

    n = min(
        len(gps),
        len(acceleration),
        len(gyro),
    )

    gps = gps[:n]
    acc = acceleration[:n]
    gyr = gyro[:n]

    features = build_health_features(
        gps,
        acc,
        gyr,
        dt=0.1,
        window=10,
    )

    normalized = (
        (
            features
            -
            mean
        )
        /
        std
    ).astype(
        np.float32
    )

    valid_frames = np.arange(
        SEQUENCE_LENGTH - 1,
        n,
        dtype=np.int64,
    )

    loader = DataLoader(
        InferenceDataset(
            normalized,
            valid_frames,
        ),
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    current_prediction = np.ones(
        n,
        dtype=np.float64,
    )

    source_future = np.ones(
        n,
        dtype=np.float64,
    )

    target_prior = np.ones(
        n,
        dtype=np.float64,
    )

    source_frame_for_target = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    model.eval()

    for batch in loader:
        x = batch[
            "feature"
        ].to(
            device
        )

        frame_ids = batch[
            "frame_id"
        ].numpy()

        output = model(
            x
        )

        current = (
            output[
                "current"
            ]
            .cpu()
            .numpy()
        )

        future = (
            output[
                "future"
            ]
            .cpu()
            .numpy()
        )

        current_prediction[
            frame_ids
        ] = current

        source_future[
            frame_ids
        ] = future

        for source, pred in zip(
            frame_ids,
            future,
        ):
            target = (
                int(source)
                +
                HORIZON
            )

            if target < n:
                target_prior[
                    target
                ] = float(
                    pred
                )

                source_frame_for_target[
                    target
                ] = int(
                    source
                )

    # Frames without H-step predictions use current-head output.
    for i in range(n):
        if (
            source_frame_for_target[i]
            <
            0
        ):
            target_prior[i] = (
                current_prediction[i]
            )

    np.savetxt(
        os.path.join(
            folder,
            "predictive_prior.txt",
        ),
        target_prior,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            folder,
            "current_prediction.txt",
        ),
        current_prediction,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            folder,
            "future_prediction_source_aligned.txt",
        ),
        source_future,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            folder,
            "prediction_source_frame.txt",
        ),
        source_frame_for_target,
        fmt="%d",
    )

    print(
        f"{scenario:24s} "
        f"prior mean="
        f"{target_prior.mean():.4f} "
        f"min="
        f"{target_prior.min():.4f} "
        f"max="
        f"{target_prior.max():.4f}"
    )


def main():
    if not os.path.exists(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            MODEL_PATH
        )

    if not os.path.exists(
        NORMALIZATION_PATH
    ):
        raise FileNotFoundError(
            NORMALIZATION_PATH
        )

    normalization = np.load(
        NORMALIZATION_PATH
    )

    mean = np.asarray(
        normalization[
            "mean"
        ],
        dtype=np.float32,
    )

    std = np.asarray(
        normalization[
            "std"
        ],
        dtype=np.float32,
    )

    std[
        std < 1e-6
    ] = 1.0

    acceleration, gyro = (
        load_imu()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    model = (
        PredictiveReliabilityModel(
            input_dim=
                len(mean),
            hidden_dim=128,
            num_layers=2,
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    print("=" * 88)
    print(
        "FROZEN H=3 MODEL "
        "MULTI-SCENARIO INFERENCE"
    )
    print(
        "Device:",
        device,
    )
    print(
        "Temporal backend:",
        model.backend,
    )
    print("=" * 88)

    for scenario in SCENARIOS:
        infer_scenario(
            model,
            device,
            mean,
            std,
            acceleration,
            gyro,
            scenario,
        )


if __name__ == "__main__":
    main()
